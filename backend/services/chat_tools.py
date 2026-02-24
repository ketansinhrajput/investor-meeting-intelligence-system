"""Deterministic structured retrieval tools for the chatbot.

All tools operate on pre-loaded in-memory run data (dict).
No disk I/O during tool execution. No LLM calls.
Results include qa_id and page refs for citation grounding.
"""

import json
from typing import Any, Optional


# Budget for adaptive text truncation per tool response
TOTAL_CHAR_BUDGET = 3000


def _truncate_adaptive(text: str, n_results: int, total_budget: int = TOTAL_CHAR_BUDGET) -> str:
    """Truncate text adaptively based on number of results sharing the budget."""
    per_item = total_budget // max(n_results, 1)
    if len(text) <= per_item:
        return text
    return text[:per_item] + "..."


def _format_result(data: Any) -> str:
    """Format tool result as a JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


# ============================================================
# Tool: get_run_metadata
# ============================================================

def get_run_metadata(run_data: dict) -> str:
    """Get basic call information: company, quarter, date, counts."""
    meta = run_data.get("meta", {})
    metadata = run_data.get("metadata", {})
    qa = run_data.get("qa", {})
    speakers = run_data.get("speakers", {})

    return _format_result({
        "company_name": meta.get("company_name"),
        "ticker_symbol": meta.get("ticker_symbol"),
        "fiscal_quarter": meta.get("fiscal_quarter"),
        "fiscal_year": meta.get("fiscal_year"),
        "call_date": meta.get("call_date"),
        "call_title": meta.get("call_title"),
        "speaker_count": len(speakers.get("speakers", {})),
        "qa_count": len(qa.get("qa_units", [])),
        "page_count": metadata.get("page_count", "unknown"),
    })


# ============================================================
# Tool: search_speakers
# ============================================================

def search_speakers(run_data: dict, role: Optional[str] = None,
                    name_query: Optional[str] = None,
                    company: Optional[str] = None) -> str:
    """Search speakers by role, name, or company. Returns up to 20 speakers."""
    speakers = run_data.get("speakers", {}).get("speakers", {})
    results = []

    for s in speakers.values():
        if role and s.get("role") != role:
            continue
        if name_query:
            q = name_query.lower()
            name_match = q in s.get("canonical_name", "").lower()
            alias_match = any(q in a.lower() for a in s.get("aliases", []))
            if not name_match and not alias_match:
                continue
        if company:
            if not s.get("company") or company.lower() not in s["company"].lower():
                continue

        results.append({
            "speaker_id": s["speaker_id"],
            "canonical_name": s["canonical_name"],
            "role": s.get("role"),
            "title": s.get("title"),
            "company": s.get("company"),
            "turn_count": s.get("turn_count"),
            "first_appearance_page": s.get("first_appearance_page"),
        })

    return _format_result({"results": results[:20], "total": len(results)})


# ============================================================
# Tool: search_qa_units
# ============================================================

def _fuzzy_name_in(candidate: str, target: str) -> bool:
    """Check if candidate name fuzzy-matches target name.

    Handles misspellings by checking edit distance on individual tokens.
    """
    if not candidate or not target:
        return False
    c_tokens = candidate.lower().split()
    t_tokens = target.lower().split()

    for ct in c_tokens:
        if len(ct) < 3:
            continue
        for tt in t_tokens:
            if len(tt) < 3:
                continue
            # Exact token substring
            if ct in tt or tt in ct:
                return True
            # Edit distance check
            if len(ct) > 2 and len(tt) > 2:
                dist = _simple_edit_distance(ct, tt)
                max_allowed = 1 if min(len(ct), len(tt)) <= 5 else 2
                if dist <= max_allowed:
                    return True
    return False


def _simple_edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _simple_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def search_qa_units(run_data: dict, keyword: Optional[str] = None,
                    questioner_name: Optional[str] = None,
                    responder_name: Optional[str] = None,
                    is_follow_up: Optional[bool] = None,
                    limit: int = 5) -> str:
    """Search Q&A exchanges by keyword, speaker, or follow-up status.

    Returns adaptively truncated summaries with qa_id and page refs.
    Uses fuzzy matching for questioner_name when exact match returns 0 results.
    """
    qa_units = run_data.get("qa", {}).get("qa_units", [])
    matched = []

    for q in qa_units:
        if questioner_name:
            if questioner_name.lower() not in q.get("questioner_name", "").lower():
                continue
        if responder_name:
            if not any(responder_name.lower() in r.lower() for r in q.get("responder_names", [])):
                continue
        if keyword:
            kw = keyword.lower()
            text = (q.get("question_text", "") + " " + q.get("response_text", "")).lower()
            if kw not in text:
                continue
        if is_follow_up is not None:
            if q.get("is_follow_up") != is_follow_up:
                continue
        matched.append(q)

    # Fuzzy fallback: if questioner_name search returned 0, retry with fuzzy matching
    if questioner_name and not matched and not keyword:
        for q in qa_units:
            if _fuzzy_name_in(questioner_name, q.get("questioner_name", "")):
                if is_follow_up is not None and q.get("is_follow_up") != is_follow_up:
                    continue
                matched.append(q)

    # Apply limit
    results = matched[:limit]
    n = len(results)

    formatted = []
    for q in results:
        formatted.append({
            "qa_id": q["qa_id"],
            "questioner_name": q.get("questioner_name"),
            "responder_names": q.get("responder_names", []),
            "question_text": _truncate_adaptive(q.get("question_text", ""), n),
            "response_text": _truncate_adaptive(q.get("response_text", ""), n),
            "is_follow_up": q.get("is_follow_up"),
            "follow_up_of": q.get("follow_up_of"),
            "start_page": q.get("start_page"),
            "end_page": q.get("end_page"),
            "sequence": q.get("sequence_in_session"),
        })

    return _format_result({
        "results": formatted,
        "total_matching": len(matched),
        "showing": n,
    })


# ============================================================
# Tool: get_qa_detail
# ============================================================

def get_qa_detail(run_data: dict, qa_id: str) -> str:
    """Get full untruncated details of a specific Q&A exchange."""
    qa_units = run_data.get("qa", {}).get("qa_units", [])
    for q in qa_units:
        if q["qa_id"] == qa_id:
            return _format_result(q)
    return _format_result({"error": f"Q&A unit '{qa_id}' not found"})


# ============================================================
# Tool: get_follow_up_chain
# ============================================================

def get_follow_up_chain(run_data: dict, qa_id: str) -> str:
    """Get the full chain of follow-up Q&As from a given starting point.

    Traverses follow_up_of links to find the root, then collects all
    Q&As in the chain ordered by sequence.
    """
    qa_units = run_data.get("qa", {}).get("qa_units", [])
    qa_map = {q["qa_id"]: q for q in qa_units}

    current = qa_map.get(qa_id)
    if not current:
        return _format_result({"error": f"Q&A unit '{qa_id}' not found"})

    # Find root of the chain
    while current.get("follow_up_of") and current["follow_up_of"] in qa_map:
        current = qa_map[current["follow_up_of"]]

    # Collect all Q&As in the chain via BFS
    chain = [current]
    visited = {current["qa_id"]}
    changed = True
    while changed:
        changed = False
        for q in qa_units:
            if q["qa_id"] not in visited and q.get("follow_up_of") in visited:
                chain.append(q)
                visited.add(q["qa_id"])
                changed = True

    chain.sort(key=lambda x: x.get("sequence_in_session", 0))

    formatted = []
    for q in chain:
        formatted.append({
            "qa_id": q["qa_id"],
            "questioner_name": q.get("questioner_name"),
            "responder_names": q.get("responder_names", []),
            "question_text": q.get("question_text", "")[:600],
            "response_text": q.get("response_text", "")[:600],
            "start_page": q.get("start_page"),
            "end_page": q.get("end_page"),
            "is_follow_up": q.get("is_follow_up"),
        })

    return _format_result({"chain_length": len(chain), "chain": formatted})


# ============================================================
# Tool: search_strategic_statements
# ============================================================

def search_strategic_statements(run_data: dict, keyword: Optional[str] = None,
                                statement_type: Optional[str] = None,
                                is_forward_looking: Optional[bool] = None,
                                speaker_name: Optional[str] = None,
                                limit: int = 10) -> str:
    """Search strategic statements (guidance, outlook, etc.)."""
    strategic = run_data.get("strategic", {})
    statements = strategic.get("statements", [])
    matched = []

    for s in statements:
        if keyword and keyword.lower() not in s.get("text", "").lower():
            continue
        if statement_type and s.get("statement_type") != statement_type:
            continue
        if is_forward_looking is not None and s.get("is_forward_looking") != is_forward_looking:
            continue
        if speaker_name and speaker_name.lower() not in s.get("speaker_name", "").lower():
            continue
        matched.append(s)

    results = matched[:limit]
    return _format_result({
        "results": results,
        "total_matching": len(matched),
        "showing": len(results),
    })


# ============================================================
# Tool: search_full_text
# ============================================================

def search_full_text(run_data: dict, keyword: str,
                     context_chars: int = 300) -> str:
    """Search raw transcript text for a keyword.

    Returns matching excerpts with page numbers. This is the fallback
    when structured filters miss due to vocabulary mismatch.
    """
    extraction = run_data.get("extraction", {})
    kw_lower = keyword.lower()
    results = []

    # Handle different extraction data structures
    pages = extraction.get("pages", [])
    if not pages:
        # Some runs store as raw_text string
        raw = extraction.get("raw_text", "")
        if raw:
            pages = [{"page_number": 0, "text": raw}]

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_number", 0)
        idx = text.lower().find(kw_lower)
        while idx != -1:
            start = max(0, idx - context_chars // 2)
            end = min(len(text), idx + len(keyword) + context_chars // 2)
            results.append({
                "page_number": page_num,
                "excerpt": text[start:end],
                "keyword_found": keyword,
            })
            if len(results) >= 5:
                break
            idx = text.lower().find(kw_lower, idx + 1)
        if len(results) >= 5:
            break

    return _format_result({
        "results": results,
        "total_matches": len(results),
        "keyword": keyword,
    })


# ============================================================
# Tool: get_raw_text_page
# ============================================================

def get_raw_text_page(run_data: dict, page_number: int) -> str:
    """Get raw transcript text for a specific page."""
    extraction = run_data.get("extraction", {})
    pages = extraction.get("pages", [])
    for page in pages:
        if page.get("page_number") == page_number:
            return _format_result({
                "page_number": page_number,
                "text": page.get("text", "")[:3000],
            })
    return _format_result({"error": f"Page {page_number} not found"})


# ============================================================
# Tool Dispatcher
# ============================================================

TOOL_REGISTRY = {
    "get_run_metadata": lambda data, params: get_run_metadata(data),
    "search_speakers": lambda data, params: search_speakers(data, **params),
    "search_qa_units": lambda data, params: search_qa_units(data, **params),
    "get_qa_detail": lambda data, params: get_qa_detail(data, **params),
    "get_follow_up_chain": lambda data, params: get_follow_up_chain(data, **params),
    "search_strategic_statements": lambda data, params: search_strategic_statements(data, **params),
    "search_full_text": lambda data, params: search_full_text(data, **params),
    "get_raw_text_page": lambda data, params: get_raw_text_page(data, **params),
}


def execute_tool(tool_name: str, params: dict, run_data: dict) -> str:
    """Execute a tool by name with given params against run data.

    Returns JSON string result or error.
    """
    handler = TOOL_REGISTRY.get(tool_name)
    if not handler:
        return _format_result({"error": f"Unknown tool: {tool_name}"})

    try:
        return handler(run_data, params)
    except Exception as e:
        return _format_result({"error": f"Tool '{tool_name}' failed: {str(e)}"})


def build_summary_evidence(run_data: dict, max_qa: int = 50) -> str:
    """Build condensed summary evidence grouped by analyst.

    Instead of raw JSON (which LLMs tend to dump verbatim), produces
    a pre-grouped text format that naturally leads to thematic synthesis.

    Format per analyst:
        ANALYST: Name (N questions)
        - Q1 [qa_XXX]: question snippet... -> response snippet...
        - Q2 [qa_XXX]: question snippet... -> response snippet...
    """
    qa_units = run_data.get("qa", {}).get("qa_units", [])
    if not qa_units:
        return "No Q&A exchanges found in this transcript."

    # Group by questioner
    by_questioner: dict[str, list[dict]] = {}
    for q in qa_units[:max_qa]:
        name = q.get("questioner_name", "Unknown")
        by_questioner.setdefault(name, []).append(q)

    # Adaptive snippet length based on total QA count
    n_total = min(len(qa_units), max_qa)
    q_snippet_len = max(80, 400 // max(n_total // 5, 1))
    r_snippet_len = max(100, 600 // max(n_total // 5, 1))

    lines = []
    lines.append(f"Q&A SESSION: {n_total} exchanges from {len(by_questioner)} analysts\n")

    for name, questions in by_questioner.items():
        lines.append(f"ANALYST: {name} ({len(questions)} question{'s' if len(questions) != 1 else ''})")
        for q in questions:
            qa_id = q.get("qa_id", "?")
            q_text = q.get("question_text", "").strip()
            r_text = q.get("response_text", "").strip()
            responders = ", ".join(q.get("responder_names", []))

            q_snip = q_text[:q_snippet_len] + ("..." if len(q_text) > q_snippet_len else "")
            r_snip = r_text[:r_snippet_len] + ("..." if len(r_text) > r_snippet_len else "")

            follow_up = " (follow-up)" if q.get("is_follow_up") else ""
            resp_note = f" [{responders}]" if responders else ""
            lines.append(f"  - [{qa_id}]{follow_up}: {q_snip}")
            lines.append(f"    Response{resp_note}: {r_snip}")
        lines.append("")  # blank line between analysts

    return "\n".join(lines)


def count_results(tool_result_json: str) -> int:
    """Count how many results a tool returned (for fallback threshold check)."""
    try:
        data = json.loads(tool_result_json)
        if "results" in data:
            return len(data["results"])
        if "chain" in data:
            return len(data["chain"])
        if "error" in data:
            return 0
        return 1  # Single result (e.g., get_run_metadata)
    except (json.JSONDecodeError, TypeError):
        return 0
