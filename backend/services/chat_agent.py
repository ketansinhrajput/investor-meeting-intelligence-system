"""2-Phase Chatbot Agent Orchestrator.

Phase 1: Tool Selection — LLM outputs JSON action (with retry on empty)
Phase 2: Grounded Synthesis — LLM answers ONLY from retrieved evidence

Layer 1 (PRIMARY): Structured deterministic retrieval
Layer 2 (FALLBACK): Vector similarity search when structured returns < threshold
"""

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from langchain_ollama import OllamaLLM

from services.chat_data_loader import generate_data_summary, get_summary_text
from services.chat_prompts import (
    build_analyst_synthesis_prompt,
    build_summary_synthesis_prompt,
    build_synthesis_prompt,
    build_tool_selection_prompt,
    build_vector_fallback_prompt,
)
from services.chat_tools import build_summary_evidence, count_results, execute_tool, search_qa_units
from services.chat_vector import index_run_data, vector_search

logger = logging.getLogger(__name__)

# Configuration
MAX_TOOL_RETRIES = 3           # Retries for empty LLM responses in Phase 1
VECTOR_FALLBACK_THRESHOLD = 2  # If structured returns < this, try vector
MAX_EVIDENCE_CHARS = 3000      # Max chars of evidence sent to synthesis
MAX_EVIDENCE_CHARS_SUMMARY = 6000  # Higher budget for summary queries
MAX_EVIDENCE_CHARS_ANALYST = 4000  # Budget for analyst-specific queries
LLM_MODEL = "gpt-oss:20b"
# LLM_MODEL = "gpt-oss:120b-cloud"
LLM_BASE_URL = "http://localhost:11434"
LLM_NUM_CTX = 16384
LLM_TEMPERATURE = 0.1

# Greeting / small-talk patterns (checked before any LLM call)
GREETING_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you", re.IGNORECASE),
    re.compile(r"^what\s+can\s+you\s+(do|help)", re.IGNORECASE),
    re.compile(r"^who\s+are\s+you", re.IGNORECASE),
    re.compile(r"^(thanks|thank\s+you|thx)\b", re.IGNORECASE),
    re.compile(r"^help$", re.IGNORECASE),
    re.compile(r"^(ok|okay|cool|great|nice|awesome|got\s+it)\s*[.!]?$", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see\s+you)\b", re.IGNORECASE),
]

# Varied greeting responses — randomly selected for naturalness
GREETING_RESPONSES = [
    (
        "Hey! I'm here to help you analyze this earnings call Q&A session.\n\n"
        "You can ask me about:\n"
        "- **Key topics and themes** discussed by analysts\n"
        "- **Specific analyst questions** (e.g., 'What did Nitin Gandhi ask?')\n"
        "- **Management responses** on margins, guidance, or risks\n"
        "- **Session overview** or summary\n\n"
        "What would you like to know?"
    ),
    (
        "Hi there! I can help you explore this transcript's Q&A session.\n\n"
        "Try asking things like:\n"
        "- \"What were the main concerns raised by analysts?\"\n"
        "- \"Summarize the Q&A session\"\n"
        "- \"What questions did [analyst name] ask?\"\n"
        "- \"Who are the management speakers?\"\n\n"
        "Go ahead!"
    ),
    (
        "Hello! Feel free to ask me about analyst questions, management responses, "
        "key themes, or anything else from this earnings call Q&A.\n\n"
        "I can search through **structured Q&A data**, identify **patterns across questions**, "
        "or look up what a **specific analyst** focused on."
    ),
    (
        "Hey! I'm ready to dig into this Q&A session with you.\n\n"
        "Some ideas to get started:\n"
        "- Ask about **recurring themes** across analyst questions\n"
        "- Look up **what a specific person asked or said**\n"
        "- Get a **high-level summary** of the session\n"
        "- Explore **follow-up questions** and their context"
    ),
]

# Thank-you specific responses
THANKS_RESPONSES = [
    "You're welcome! Let me know if you have more questions about the Q&A session.",
    "Happy to help! Feel free to ask anything else about this transcript.",
    "Glad I could help! I'm here if you need more analysis.",
    "Anytime! If you want to dig deeper into any topic or analyst, just ask.",
    "No problem! There's plenty more to explore in this transcript if you're curious.",
    "You're welcome! I can also help with summaries, follow-up chains, or specific speakers.",
]

# Summary / overview request patterns
SUMMARY_PATTERNS = [
    re.compile(r"summar", re.IGNORECASE),
    re.compile(r"overview", re.IGNORECASE),
    re.compile(r"main\s+(topics|themes|points)", re.IGNORECASE),
    re.compile(r"key\s+(topics|themes|takeaways|highlights)", re.IGNORECASE),
    re.compile(r"what\s+.*(discussed|covered|talked\s+about)", re.IGNORECASE),
    re.compile(r"high.level", re.IGNORECASE),
    re.compile(r"(all|major)\s+(questions|concerns|themes)", re.IGNORECASE),
    re.compile(r"brief\s+(me|us|overview)", re.IGNORECASE),
]

# Analyst-specific question patterns (bypass tool selection, use structured QA directly)
ANALYST_QUESTION_PATTERNS = [
    re.compile(r"what\s+did\s+(.+?)\s+(ask|say|question|raise|mention|focus)", re.IGNORECASE),
    re.compile(r"what\s+questions?\s+did\s+(.+?)\s+(ask|raise|have)", re.IGNORECASE),
    re.compile(r"(.+?)'s\s+questions?", re.IGNORECASE),
    re.compile(r"questions?\s+(from|by|asked\s+by)\s+(.+)", re.IGNORECASE),
    re.compile(r"what\s+was\s+(.+?)\s+(asking|focused|concerned)", re.IGNORECASE),
    # Passive: "what questions are asked by X"
    re.compile(r"what\s+(?:questions?\s+)?(?:are|were)\s+asked\s+by\s+(.+)", re.IGNORECASE),
    # "what was QA of X", "QA of X", "tell me about QA of X"
    re.compile(r"(?:what\s+(?:was|is|are)\s+)?(?:the\s+)?QA\s+(?:of|for|from|by)\s+(.+)", re.IGNORECASE),
    # "tell me about questions asked by X" / "tell me about X's questions"
    re.compile(r"tell\s+me\s+about\s+(?:questions?\s+(?:asked|from|by)\s+)?(.+?)(?:'s\s+questions?)?$", re.IGNORECASE),
    # "questions of X" / "QA by X"
    re.compile(r"(?:questions?|QA)\s+(?:of|by|from)\s+(.+)", re.IGNORECASE),
    # "what X asked" (name first)
    re.compile(r"what\s+(.+?)\s+asked", re.IGNORECASE),
]

# Words to exclude from analyst name extraction (common verbs/prepositions captured by regex)
_ANALYST_NAME_STOPWORDS = {
    "ask", "say", "question", "raise", "mention", "focus",
    "have", "from", "by", "asking", "focused", "concerned",
    "asked", "the", "all", "about", "questions", "qa",
    "tell", "me", "was", "is", "are", "were", "did",
    "of", "for", "in", "on", "to", "and", "or", "a", "an",
}


def _fuzzy_name_match(candidate: str, known_names: list[str], threshold: float = 0.6) -> Optional[str]:
    """Fuzzy match a candidate name against known names.

    Uses token-level matching: checks if any token in the candidate
    is close enough to any token in a known name (edit distance).
    Returns the best matching known name or None.
    """
    candidate_lower = candidate.lower().strip()
    if not candidate_lower:
        return None

    candidate_tokens = candidate_lower.split()
    best_match = None
    best_score = 0.0

    for name in known_names:
        name_lower = name.lower()
        name_tokens = name_lower.split()

        # Exact substring match (already handled elsewhere, but include for scoring)
        if candidate_lower in name_lower or name_lower in candidate_lower:
            return name

        # Token-level matching
        token_matches = 0
        for ct in candidate_tokens:
            if len(ct) < 3:
                continue
            for nt in name_tokens:
                if len(nt) < 3:
                    continue
                # Check edit distance (allow 1-2 edits based on length)
                dist = _edit_distance(ct, nt)
                max_allowed = 1 if len(ct) <= 5 else 2
                if dist <= max_allowed:
                    token_matches += 1
                    break

        if token_matches > 0:
            score = token_matches / max(len(candidate_tokens), 1)
            if score > best_score:
                best_score = score
                best_match = name

    return best_match if best_score >= threshold else None


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# History / meta-question patterns (answer from conversation history, not retrieval)
HISTORY_PATTERNS = [
    re.compile(r"what\s+(?:did\s+)?I\s+ask", re.IGNORECASE),
    re.compile(r"(?:my|our)\s+(previous|last|earlier)\s+question", re.IGNORECASE),
    re.compile(r"what\s+(?:did\s+)?(?:we|I)\s+(?:discuss|talk)", re.IGNORECASE),
    re.compile(r"(?:previous|last|earlier)\s+(?:question|message|conversation)", re.IGNORECASE),
    re.compile(r"what\s+(?:was|were)\s+(?:my|our)\s+(?:question|message)", re.IGNORECASE),
    re.compile(r"what\s+I\s+(?:said|asked|wrote)\s+(?:before|earlier|previously)", re.IGNORECASE),
    re.compile(r"repeat\s+(?:my|the)\s+(?:last|previous)\s+question", re.IGNORECASE),
]


def _is_history_question(question: str) -> bool:
    """Check if the question is asking about conversation history."""
    q = question.strip()
    return any(p.search(q) for p in HISTORY_PATTERNS)


def _answer_from_history(question: str, history: list[dict], summary: dict) -> Optional[str]:
    """Try to answer a meta-question from conversation history.

    Returns a formatted answer or None if history is empty.
    """
    if not history:
        suggestions = _build_suggested_questions(summary)
        return (
            "This is the start of our conversation -- there are no previous messages yet."
            + suggestions
        )

    # Collect previous user messages
    user_messages = [m for m in history if m.get("role") == "user"]
    if not user_messages:
        return "I don't see any previous questions from you in this conversation."

    # Format the last few user messages
    recent = user_messages[-5:]  # show last 5
    lines = []
    for i, m in enumerate(recent, 1):
        lines.append(f"{i}. \"{m['content']}\"")

    count_text = f"your last {len(recent)}" if len(recent) < len(user_messages) else "all your"
    return (
        f"Here are {count_text} questions in this conversation:\n\n"
        + "\n".join(lines)
    )


def _is_greeting(question: str) -> bool:
    """Check if the question is a greeting or small-talk (not analytical)."""
    q = question.strip()
    return any(p.search(q) for p in GREETING_PATTERNS)


def _is_thanks(question: str) -> bool:
    """Check if user is saying thanks."""
    q = question.strip()
    return bool(re.match(r"^(thanks|thank\s+you|thx)\b", q, re.IGNORECASE))


def _get_greeting_response() -> str:
    """Return a randomly selected greeting response for variety."""
    return random.choice(GREETING_RESPONSES)


def _get_thanks_response() -> str:
    """Return a randomly selected thanks response."""
    return random.choice(THANKS_RESPONSES)


def _is_summary_request(question: str) -> bool:
    """Check if the question asks for a summary or overview of the Q&A session."""
    q = question.strip()
    return any(p.search(q) for p in SUMMARY_PATTERNS)


def _extract_analyst_name(question: str, known_names: list[str]) -> Optional[str]:
    """Try to extract an analyst name from the question.

    Steps:
    1. Regex patterns extract a candidate name string
    2. Exact substring match against known names
    3. Fuzzy match (edit distance) against known names
    4. Return best match or raw candidate as last resort
    """
    q = question.strip()

    # Try each pattern to extract a candidate name
    candidate = None
    for pattern in ANALYST_QUESTION_PATTERNS:
        m = pattern.search(q)
        if m:
            # Different patterns capture the name in different groups
            for g in m.groups():
                if g and len(g) > 2 and g.lower().strip() not in _ANALYST_NAME_STOPWORDS:
                    cleaned = g.strip().rstrip("?.,!").strip()
                    # Skip if all tokens are stopwords
                    tokens = cleaned.lower().split()
                    non_stop = [t for t in tokens if t not in _ANALYST_NAME_STOPWORDS and len(t) > 1]
                    if non_stop:
                        candidate = cleaned
                        break
            if candidate:
                break

    if not candidate:
        return None

    # Clean stopwords from candidate (e.g. "Nirav questions" -> "Nirav")
    cleaned_tokens = [t for t in candidate.split()
                      if t.lower() not in _ANALYST_NAME_STOPWORDS and len(t) > 1]
    if cleaned_tokens:
        candidate = " ".join(cleaned_tokens)
    else:
        return None

    logger.info(f"Analyst name candidate extracted: '{candidate}'")

    # Step 1: Exact substring match against known names
    candidate_lower = candidate.lower()
    for name in known_names:
        if candidate_lower in name.lower() or name.lower() in candidate_lower:
            logger.info(f"Exact match: '{candidate}' -> '{name}'")
            return name

    # Step 2: Fuzzy match against known names (handles misspellings)
    fuzzy_match = _fuzzy_name_match(candidate, known_names)
    if fuzzy_match:
        logger.info(f"Fuzzy match: '{candidate}' -> '{fuzzy_match}'")
        return fuzzy_match

    # Step 3: Return raw candidate (search_qa_units does partial match internally)
    logger.info(f"No known name match for '{candidate}', using raw candidate")
    return candidate


def _build_suggested_questions(summary: dict) -> str:
    """Build a friendly nudge when a response is weak or fails.

    Returns a generic, warm message about what the user can explore
    rather than listing specific questions.
    """
    company = summary.get("company", "this company")
    qa_count = summary.get("qa_count", 0)
    parts = []
    if qa_count > 0:
        parts.append(f"analyst Q&A exchanges ({qa_count} available)")
    parts.append("speaker information")
    parts.append("key topics and themes")
    available = ", ".join(parts)
    return (
        f"\n\nI can help you explore this earnings call transcript -- "
        f"including {available}. "
        f"Try asking about a specific topic, analyst, or request a summary!"
    )


@dataclass
class ChatResponse:
    """Response from the chat agent."""
    answer: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    retrieval_source: str = "none"  # "structured", "vector", "both"
    total_time_seconds: float = 0.0
    model: str = LLM_MODEL
    disclaimer: str = ""


def _create_llm() -> OllamaLLM:
    """Create the OllamaLLM instance for text generation."""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        num_predict=2048,
        streaming=False,
    )


def _parse_json_from_text(text: str) -> Optional[dict]:
    """Extract and parse a JSON object from LLM text output.

    Handles cases where the model outputs text around the JSON.
    """
    text = text.strip()
    if not text or "{" not in text:
        return None

    try:
        # Find the first { and last } to extract JSON
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None


def _extract_citations(answer: str, run_data: dict) -> list[dict]:
    """Extract citation references from the answer text.

    Looks for patterns like [qa_000], [page_5], [speaker_002].
    Maps each to a human-readable label.
    """
    citations = []
    seen = set()

    # Q&A citations: [qa_000], [qa_001], 【qa_000】, etc.
    qa_pattern = re.compile(r"[\[【]qa_(\d+)[\]】]")
    for match in qa_pattern.finditer(answer):
        qa_id = f"qa_{match.group(1)}"
        if qa_id in seen:
            continue
        seen.add(qa_id)

        # Look up questioner name for label
        label = qa_id
        qa_units = run_data.get("qa", {}).get("qa_units", [])
        for q in qa_units:
            if q["qa_id"] == qa_id:
                questioner = q.get("questioner_name", "Unknown")
                label = f"Q&A #{match.group(1)} ({questioner})"
                break

        citations.append({"type": "qa", "ref_id": qa_id, "label": label})

    # Page citations: [page_5], [page_12], 【page_5】, etc.
    page_pattern = re.compile(r"[\[【]page_(\d+)[\]】]")
    for match in page_pattern.finditer(answer):
        page_ref = f"page_{match.group(1)}"
        if page_ref in seen:
            continue
        seen.add(page_ref)
        citations.append({"type": "page", "ref_id": match.group(1), "label": f"Page {match.group(1)}"})

    # Speaker citations: [speaker_000], 【speaker_000】, etc.
    speaker_pattern = re.compile(r"[\[【]speaker_(\d+)[\]】]")
    for match in speaker_pattern.finditer(answer):
        speaker_id = f"speaker_{match.group(1)}"
        if speaker_id in seen:
            continue
        seen.add(speaker_id)

        label = speaker_id
        speakers = run_data.get("speakers", {}).get("speakers", {})
        if speaker_id in speakers:
            label = speakers[speaker_id].get("canonical_name", speaker_id)

        citations.append({"type": "speaker", "ref_id": speaker_id, "label": label})

    return citations


# ============================================================
# Phase 1: Tool Selection
# ============================================================

def _phase1_select_tool(llm: OllamaLLM, summary: dict, question: str) -> Optional[dict]:
    """Phase 1: Ask the LLM to select a retrieval tool.

    Retries up to MAX_TOOL_RETRIES times on empty responses.
    Returns parsed JSON action or None if all retries fail.
    """
    prompt = build_tool_selection_prompt(
        company=summary["company"],
        quarter=summary["quarter"],
        year=summary["year"],
        speaker_count=summary["speaker_count"],
        qa_count=summary["qa_count"],
        speaker_names=", ".join(summary["management_names"] + summary["analyst_names"]),
        user_question=question,
    )

    for attempt in range(MAX_TOOL_RETRIES):
        try:
            response = llm.invoke(prompt).strip()
            if not response:
                logger.warning(f"Phase 1 attempt {attempt + 1}: empty response, retrying")
                continue

            parsed = _parse_json_from_text(response)
            if parsed and "action" in parsed:
                return parsed

            # Try to salvage: if the model output just a tool name
            if parsed and "function" in parsed:
                return {
                    "action": "structured_search",
                    "tool": parsed["function"],
                    "params": parsed.get("arguments", {}),
                }

            logger.warning(f"Phase 1 attempt {attempt + 1}: invalid JSON: {response[:200]}")

        except Exception as e:
            logger.error(f"Phase 1 attempt {attempt + 1} error: {e}")

    return None


# ============================================================
# Phase 2: Grounded Synthesis
# ============================================================

def _phase2_synthesize(llm: OllamaLLM, summary: dict, question: str,
                       evidence: str, is_vector: bool = False) -> str:
    """Phase 2: Synthesize an answer from retrieved evidence.

    Uses strict grounding rules to prevent hallucination.
    """
    if is_vector:
        prompt = build_vector_fallback_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )
    else:
        prompt = build_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )

    try:
        response = llm.invoke(prompt).strip()
        if not response:
            return "The model did not generate a response. Please try rephrasing your question."
        return response
    except Exception as e:
        logger.error(f"Phase 2 synthesis error: {e}")
        return f"An error occurred while generating the answer: {str(e)}"


# ============================================================
# Main Agent Entry Point
# ============================================================

def chat(question: str, run_data: dict, history: Optional[list[dict]] = None) -> ChatResponse:
    """Process a user question against run data.

    2-phase approach:
    1. LLM selects retrieval tool -> deterministic execution
    2. LLM synthesizes answer from evidence with strict grounding

    If structured retrieval returns < threshold, falls back to vector search.
    """
    start_time = time.time()
    summary = generate_data_summary(run_data)
    run_id = run_data.get("run_id", "unknown")

    # Pre-check: Thanks (no LLM needed, specific response)
    if _is_thanks(question):
        logger.info("Thanks detected, returning acknowledgment")
        return ChatResponse(
            answer=_get_thanks_response(),
            retrieval_source="none",
            total_time_seconds=round(time.time() - start_time, 2),
        )

    # Pre-check: Greeting / small-talk (no LLM needed)
    if _is_greeting(question):
        logger.info("Greeting detected, returning varied response")
        return ChatResponse(
            answer=_get_greeting_response(),
            retrieval_source="none",
            total_time_seconds=round(time.time() - start_time, 2),
        )

    # Pre-check: History / meta-question (answer from conversation history)
    if _is_history_question(question):
        logger.info("History question detected, answering from conversation context")
        answer = _answer_from_history(question, history or [], summary)
        return ChatResponse(
            answer=answer,
            retrieval_source="none",
            total_time_seconds=round(time.time() - start_time, 2),
        )

    # Pre-check: Summary / overview request (bypass Phase 1, fetch all Q&A)
    if _is_summary_request(question):
        logger.info("Summary request detected, building condensed summary evidence")
        llm = _create_llm()
        evidence = build_summary_evidence(run_data, max_qa=50)
        tool_calls = [{"tool": "build_summary_evidence", "params": {"max_qa": 50}}]

        if len(evidence) > MAX_EVIDENCE_CHARS_SUMMARY:
            evidence = evidence[:MAX_EVIDENCE_CHARS_SUMMARY] + "\n... (truncated)"

        prompt = build_summary_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )
        try:
            answer = llm.invoke(prompt).strip()
            if not answer:
                suggestions = _build_suggested_questions(summary)
                answer = "The model did not generate a summary. Please try again." + suggestions
        except Exception as e:
            logger.error(f"Summary synthesis error: {e}")
            answer = f"An error occurred while generating the summary: {str(e)}"

        citations = _extract_citations(answer, run_data)
        total_time = time.time() - start_time
        logger.info(f"Summary completed in {total_time:.1f}s: {len(citations)} citations")

        return ChatResponse(
            answer=answer,
            citations=citations,
            tool_calls=tool_calls,
            retrieval_source="structured",
            total_time_seconds=round(total_time, 2),
            model=LLM_MODEL,
            disclaimer="Summary based on all Q&A exchanges in the transcript.",
        )

    # Pre-check: Analyst-specific question (bypass Phase 1, use structured QA directly)
    all_questioner_names = summary.get("unique_questioners", [])
    analyst_name = _extract_analyst_name(question, all_questioner_names)
    if analyst_name:
        logger.info(f"Analyst question detected for: {analyst_name}")
        llm = _create_llm()
        evidence = search_qa_units(run_data, questioner_name=analyst_name, limit=20)
        result_count = count_results(evidence)
        tool_calls = [{"tool": "search_qa_units", "params": {"questioner_name": analyst_name, "limit": 20}}]
        retrieval_source = "structured"

        # If no structured results, try vector search as fallback
        if result_count == 0:
            logger.info(f"No structured QA found for '{analyst_name}', trying vector search")
            try:
                index_run_data(run_id, run_data)
                evidence = vector_search(run_id, f"questions by {analyst_name}", n_results=5)
                retrieval_source = "vector"
                tool_calls.append({"tool": "vector_search", "params": {"query": f"questions by {analyst_name}"}})
            except Exception as e:
                logger.warning(f"Vector fallback failed for analyst query: {e}")

        if len(evidence) > MAX_EVIDENCE_CHARS_ANALYST:
            evidence = evidence[:MAX_EVIDENCE_CHARS_ANALYST] + "\n... (truncated)"

        prompt = build_analyst_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            analyst_name=analyst_name,
            user_question=question,
            evidence=evidence,
        )
        try:
            answer = llm.invoke(prompt).strip()
            if not answer:
                suggestions = _build_suggested_questions(summary)
                answer = f"I found Q&A data for {analyst_name} but the model did not generate a response. Please try again." + suggestions
        except Exception as e:
            logger.error(f"Analyst synthesis error: {e}")
            answer = f"An error occurred: {str(e)}"

        citations = _extract_citations(answer, run_data)
        total_time = time.time() - start_time
        logger.info(f"Analyst query completed in {total_time:.1f}s: {len(citations)} citations")

        return ChatResponse(
            answer=answer,
            citations=citations,
            tool_calls=tool_calls,
            retrieval_source=retrieval_source,
            total_time_seconds=round(total_time, 2),
            model=LLM_MODEL,
            disclaimer=f"Answer based on structured Q&A data for {analyst_name}.",
        )

    llm = _create_llm()

    # Phase 1: Tool Selection
    logger.info(f"Phase 1: Selecting tool for question: {question[:100]}")
    action = _phase1_select_tool(llm, summary, question)

    tool_calls = []
    retrieval_source = "none"
    evidence = ""

    if action is None:
        # Phase 1 completely failed — MANDATORY vector fallback
        logger.warning("Phase 1 failed after retries, MANDATORY vector fallback")
        try:
            index_run_data(run_id, run_data)
            evidence = vector_search(run_id, question, n_results=5)
            retrieval_source = "vector"
            tool_calls.append({"tool": "vector_search", "params": {"query": question}})
        except Exception as e:
            logger.error(f"Vector fallback also failed: {e}")
            suggestions = _build_suggested_questions(summary)
            return ChatResponse(
                answer="I was unable to process your question. Please try rephrasing it." + suggestions,
                total_time_seconds=time.time() - start_time,
                disclaimer="Both tool selection and vector search failed.",
            )
    else:
        # Execute the selected tool
        tool_name = action.get("tool", "")
        params = action.get("params", {})
        tool_calls.append({"tool": tool_name, "params": params})

        logger.info(f"Executing tool: {tool_name}({json.dumps(params)})")
        evidence = execute_tool(tool_name, params, run_data)
        result_count = count_results(evidence)
        retrieval_source = "structured"

        logger.info(f"Structured search returned {result_count} results")

        # MANDATORY vector fallback when structured results are weak
        if result_count < VECTOR_FALLBACK_THRESHOLD:
            logger.info(f"Structured returned {result_count} results (< {VECTOR_FALLBACK_THRESHOLD}), "
                       f"MANDATORY vector fallback")
            try:
                index_run_data(run_id, run_data)
                vector_evidence = vector_search(run_id, question, n_results=5)
                vector_count = count_results(vector_evidence)

                if vector_count > 0:
                    if result_count > 0:
                        # Combine both results
                        evidence = f"STRUCTURED SEARCH RESULTS:\n{evidence}\n\nVECTOR SEARCH RESULTS (semantic matches):\n{vector_evidence}"
                        retrieval_source = "both"
                    else:
                        # Structured returned nothing — use vector only
                        evidence = vector_evidence
                        retrieval_source = "vector"
                    tool_calls.append({"tool": "vector_search", "params": {"query": question}})
                    logger.info(f"Vector search added {vector_count} results")

            except Exception as e:
                logger.warning(f"Vector fallback failed (continuing with structured results): {e}")

    # Trim evidence to budget
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS] + "\n... (truncated)"

    # Phase 2: Grounded Synthesis
    logger.info(f"Phase 2: Synthesizing answer from {len(evidence)} chars of evidence")
    is_vector = retrieval_source in ("vector", "both")
    answer = _phase2_synthesize(llm, summary, question, evidence, is_vector=is_vector)

    # Append suggested questions if the answer indicates failure/weakness
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in [
        "did not generate a response",
        "does not contain sufficient",
        "does not contain enough",
        "an error occurred",
        "unable to process",
    ]):
        answer += _build_suggested_questions(summary)

    # Extract citations
    citations = _extract_citations(answer, run_data)

    total_time = time.time() - start_time
    logger.info(f"Chat completed in {total_time:.1f}s: {len(citations)} citations, source={retrieval_source}")

    # Build disclaimer
    disclaimer = ""
    if retrieval_source == "vector":
        disclaimer = "Answer based on semantic search results. Citations may be approximate."
    elif retrieval_source == "both":
        disclaimer = "Answer combines structured and semantic search results."
    elif retrieval_source == "structured":
        disclaimer = "Answer based on structured data extraction from the transcript."

    return ChatResponse(
        answer=answer,
        citations=citations,
        tool_calls=tool_calls,
        retrieval_source=retrieval_source,
        total_time_seconds=round(total_time, 2),
        model=LLM_MODEL,
        disclaimer=disclaimer,
    )


# ============================================================
# Streaming Entry Point (SSE)
# ============================================================

def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def chat_stream(question: str, run_data: dict, history: Optional[list[dict]] = None) -> Generator[str, None, None]:
    """Stream a chat response as SSE events.

    Yields SSE-formatted strings:
      event: metadata  — tool_calls, retrieval_source (sent before synthesis)
      event: token     — incremental text chunks during Phase 2
      event: done      — citations, timing, disclaimer (sent after synthesis)
    """
    start_time = time.time()
    summary = generate_data_summary(run_data)
    run_id = run_data.get("run_id", "unknown")

    # --- Thanks (no LLM, no streaming needed) ---
    if _is_thanks(question):
        yield _sse_event("metadata", {"tool_calls": [], "retrieval_source": "none"})
        yield _sse_event("token", {"text": _get_thanks_response()})
        yield _sse_event("done", {
            "citations": [],
            "total_time_seconds": round(time.time() - start_time, 2),
            "disclaimer": "",
            "model": LLM_MODEL,
        })
        return

    # --- Greeting (no LLM, no streaming needed) ---
    if _is_greeting(question):
        yield _sse_event("metadata", {"tool_calls": [], "retrieval_source": "none"})
        yield _sse_event("token", {"text": _get_greeting_response()})
        yield _sse_event("done", {
            "citations": [],
            "total_time_seconds": round(time.time() - start_time, 2),
            "disclaimer": "",
            "model": LLM_MODEL,
        })
        return

    # --- History question ---
    if _is_history_question(question):
        answer = _answer_from_history(question, history or [], summary)
        yield _sse_event("metadata", {"tool_calls": [], "retrieval_source": "none"})
        yield _sse_event("token", {"text": answer})
        yield _sse_event("done", {
            "citations": [],
            "total_time_seconds": round(time.time() - start_time, 2),
            "disclaimer": "",
            "model": LLM_MODEL,
        })
        return

    # --- Summary request ---
    if _is_summary_request(question):
        llm = _create_llm()
        evidence = build_summary_evidence(run_data, max_qa=50)
        tool_calls = [{"tool": "build_summary_evidence", "params": {"max_qa": 50}}]

        if len(evidence) > MAX_EVIDENCE_CHARS_SUMMARY:
            evidence = evidence[:MAX_EVIDENCE_CHARS_SUMMARY] + "\n... (truncated)"

        yield _sse_event("metadata", {"tool_calls": tool_calls, "retrieval_source": "structured"})

        prompt = build_summary_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )

        # Stream synthesis
        full_answer = ""
        try:
            for chunk in llm.stream(prompt):
                text = chunk if isinstance(chunk, str) else str(chunk)
                if text:
                    full_answer += text
                    yield _sse_event("token", {"text": text})
        except Exception as e:
            logger.error(f"Summary stream error: {e}")
            full_answer = f"An error occurred: {str(e)}"
            yield _sse_event("token", {"text": full_answer})

        if not full_answer:
            suggestions = _build_suggested_questions(summary)
            full_answer = "The model did not generate a summary. Please try again." + suggestions
            yield _sse_event("token", {"text": full_answer})

        citations = _extract_citations(full_answer, run_data)
        yield _sse_event("done", {
            "citations": citations,
            "total_time_seconds": round(time.time() - start_time, 2),
            "disclaimer": "Summary based on all Q&A exchanges in the transcript.",
            "model": LLM_MODEL,
        })
        return

    # --- Analyst-specific question ---
    all_questioner_names = summary.get("unique_questioners", [])
    analyst_name = _extract_analyst_name(question, all_questioner_names)
    if analyst_name:
        logger.info(f"[stream] Analyst question detected for: {analyst_name}")
        llm = _create_llm()
        evidence = search_qa_units(run_data, questioner_name=analyst_name, limit=20)
        result_count = count_results(evidence)
        tool_calls = [{"tool": "search_qa_units", "params": {"questioner_name": analyst_name, "limit": 20}}]
        retrieval_source = "structured"

        if result_count == 0:
            try:
                index_run_data(run_id, run_data)
                evidence = vector_search(run_id, f"questions by {analyst_name}", n_results=5)
                retrieval_source = "vector"
                tool_calls.append({"tool": "vector_search", "params": {"query": f"questions by {analyst_name}"}})
            except Exception as e:
                logger.warning(f"[stream] Vector fallback failed for analyst query: {e}")

        if len(evidence) > MAX_EVIDENCE_CHARS_ANALYST:
            evidence = evidence[:MAX_EVIDENCE_CHARS_ANALYST] + "\n... (truncated)"

        yield _sse_event("metadata", {"tool_calls": tool_calls, "retrieval_source": retrieval_source})

        prompt = build_analyst_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            analyst_name=analyst_name,
            user_question=question,
            evidence=evidence,
        )

        full_answer = ""
        try:
            for chunk in llm.stream(prompt):
                text = chunk if isinstance(chunk, str) else str(chunk)
                if text:
                    full_answer += text
                    yield _sse_event("token", {"text": text})
        except Exception as e:
            logger.error(f"[stream] Analyst stream error: {e}")
            full_answer = f"An error occurred: {str(e)}"
            yield _sse_event("token", {"text": full_answer})

        if not full_answer:
            suggestions = _build_suggested_questions(summary)
            full_answer = f"I found Q&A data for {analyst_name} but the model did not generate a response. Please try again." + suggestions
            yield _sse_event("token", {"text": full_answer})

        citations = _extract_citations(full_answer, run_data)
        yield _sse_event("done", {
            "citations": citations,
            "total_time_seconds": round(time.time() - start_time, 2),
            "disclaimer": f"Answer based on structured Q&A data for {analyst_name}.",
            "model": LLM_MODEL,
        })
        return

    # --- Standard 2-phase flow ---
    llm = _create_llm()

    # Phase 1: Tool Selection (synchronous - needs full JSON)
    logger.info(f"[stream] Phase 1: Selecting tool for: {question[:100]}")
    action = _phase1_select_tool(llm, summary, question)

    tool_calls = []
    retrieval_source = "none"
    evidence = ""

    if action is None:
        # Phase 1 completely failed — MANDATORY vector fallback
        logger.warning("[stream] Phase 1 failed, MANDATORY vector fallback")
        try:
            index_run_data(run_id, run_data)
            evidence = vector_search(run_id, question, n_results=5)
            retrieval_source = "vector"
            tool_calls.append({"tool": "vector_search", "params": {"query": question}})
        except Exception as e:
            logger.error(f"[stream] Vector fallback failed: {e}")
            suggestions = _build_suggested_questions(summary)
            yield _sse_event("metadata", {"tool_calls": [], "retrieval_source": "none"})
            yield _sse_event("token", {"text": "I was unable to process your question. Please try rephrasing it." + suggestions})
            yield _sse_event("done", {
                "citations": [],
                "total_time_seconds": round(time.time() - start_time, 2),
                "disclaimer": "Both tool selection and vector search failed.",
                "model": LLM_MODEL,
            })
            return
    else:
        tool_name = action.get("tool", "")
        params = action.get("params", {})
        tool_calls.append({"tool": tool_name, "params": params})

        logger.info(f"[stream] Executing tool: {tool_name}({json.dumps(params)})")
        evidence = execute_tool(tool_name, params, run_data)
        result_count = count_results(evidence)
        retrieval_source = "structured"

        # MANDATORY vector fallback when structured results are weak
        if result_count < VECTOR_FALLBACK_THRESHOLD:
            logger.info(f"[stream] Structured returned {result_count} results (< {VECTOR_FALLBACK_THRESHOLD}), "
                       f"MANDATORY vector fallback")
            try:
                index_run_data(run_id, run_data)
                vector_evidence = vector_search(run_id, question, n_results=5)
                vector_count = count_results(vector_evidence)
                if vector_count > 0:
                    if result_count > 0:
                        evidence = f"STRUCTURED SEARCH RESULTS:\n{evidence}\n\nVECTOR SEARCH RESULTS (semantic matches):\n{vector_evidence}"
                        retrieval_source = "both"
                    else:
                        evidence = vector_evidence
                        retrieval_source = "vector"
                    tool_calls.append({"tool": "vector_search", "params": {"query": question}})
                    logger.info(f"[stream] Vector search added {vector_count} results")
            except Exception as e:
                logger.warning(f"[stream] Vector fallback failed: {e}")

    # Send metadata before streaming synthesis
    yield _sse_event("metadata", {"tool_calls": tool_calls, "retrieval_source": retrieval_source})

    # Trim evidence
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence = evidence[:MAX_EVIDENCE_CHARS] + "\n... (truncated)"

    # Phase 2: Stream synthesis
    is_vector = retrieval_source in ("vector", "both")
    if is_vector:
        prompt = build_vector_fallback_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )
    else:
        prompt = build_synthesis_prompt(
            company=summary["company"],
            quarter=summary["quarter"],
            year=summary["year"],
            user_question=question,
            evidence=evidence,
        )

    full_answer = ""
    try:
        for chunk in llm.stream(prompt):
            text = chunk if isinstance(chunk, str) else str(chunk)
            if text:
                full_answer += text
                yield _sse_event("token", {"text": text})
    except Exception as e:
        logger.error(f"[stream] Phase 2 error: {e}")
        full_answer = f"An error occurred while generating the answer: {str(e)}"
        yield _sse_event("token", {"text": full_answer})

    if not full_answer:
        suggestions = _build_suggested_questions(summary)
        full_answer = "The model did not generate a response. Please try rephrasing your question." + suggestions
        yield _sse_event("token", {"text": full_answer})

    # Extract citations from complete answer and send done event
    citations = _extract_citations(full_answer, run_data)
    total_time = time.time() - start_time

    disclaimer = ""
    if retrieval_source == "vector":
        disclaimer = "Answer based on semantic search results. Citations may be approximate."
    elif retrieval_source == "both":
        disclaimer = "Answer combines structured and semantic search results."
    elif retrieval_source == "structured":
        disclaimer = "Answer based on structured data extraction from the transcript."

    yield _sse_event("done", {
        "citations": citations,
        "total_time_seconds": round(total_time, 2),
        "disclaimer": disclaimer,
        "model": LLM_MODEL,
    })
