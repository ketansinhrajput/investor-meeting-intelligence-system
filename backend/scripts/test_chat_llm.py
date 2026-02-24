"""
Phase 0: LLM Validation Script for Chatbot Agent

Tests gpt-oss:20b (or any Ollama model) for tool calling capability,
response quality, and timing. Tests both native ChatOllama tool calling
and prompt-based tool calling approaches.

Usage:
    cd backend
    python scripts/test_chat_llm.py [--run-id RUN_ID] [--model MODEL_NAME] [--num-ctx NUM_CTX]

Requirements:
    pip install langchain-ollama langchain-core
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Data Loading
# ============================================================

def load_run_data(run_id: str) -> dict:
    """Load all run data into memory."""
    runs_dir = Path(__file__).resolve().parent.parent / "data" / "runs"
    run_dir = runs_dir / run_id

    if not run_dir.exists():
        # Try partial match
        matches = [d for d in runs_dir.iterdir() if d.name.startswith(run_id)]
        if matches:
            run_dir = matches[0]
            run_id = run_dir.name
        else:
            raise FileNotFoundError(f"Run not found: {run_id}")

    data = {"run_id": run_id}
    file_map = {
        "metadata": "metadata.json",
        "meta": "stage_metadata_result.json",
        "speakers": "stage_speakers_result.json",
        "qa": "stage_qa_result.json",
        "strategic": "stage_strategic_result.json",
        "extraction": "stage_extraction_result.json",
    }

    for key, filename in file_map.items():
        filepath = run_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
        else:
            data[key] = {}

    return data


def generate_data_summary(data: dict) -> str:
    """Generate a human-readable summary of the run data for the system prompt."""
    meta = data.get("meta", {})
    speakers_data = data.get("speakers", {})
    qa_data = data.get("qa", {})
    strategic_data = data.get("strategic", {})

    speakers = speakers_data.get("speakers", {})
    management = [s["canonical_name"] for s in speakers.values() if s.get("role") == "management"]
    analysts = [s["canonical_name"] for s in speakers.values() if s.get("role") == "analyst"]

    qa_units = qa_data.get("qa_units", [])
    follow_ups = [q for q in qa_units if q.get("is_follow_up")]

    # Extract common keywords from Q&A text for topic hints
    all_text = " ".join(q.get("question_text", "") + " " + q.get("response_text", "") for q in qa_units).lower()

    summary = f"""Company: {meta.get('company_name', 'Unknown')} ({meta.get('ticker_symbol', '?')})
Call: {meta.get('fiscal_quarter', '?')} {meta.get('fiscal_year', '?')}, Date: {meta.get('call_date', '?')}
Speakers: {len(speakers)} total
  Management: {', '.join(management) if management else 'None identified'}
  Analysts: {', '.join(analysts) if analysts else 'None identified'}
Q&A Units: {len(qa_units)} total, {len(follow_ups)} follow-ups
Strategic Statements: {strategic_data.get('total_statements', 0)}
Pages: {data.get('metadata', {}).get('page_count', '?')}"""

    return summary


# ============================================================
# Tool Definitions (for native tool calling test)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "name": "get_run_metadata",
        "description": "Get basic information about this earnings call: company name, ticker, quarter, year, date, and counts of speakers/Q&As.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_speakers",
        "description": "Search for speakers in this earnings call by role (management/analyst/moderator) or name.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "enum": ["management", "analyst", "moderator", "unknown"], "description": "Filter by speaker role"},
                "name_query": {"type": "string", "description": "Search by name (case-insensitive substring match)"},
                "company": {"type": "string", "description": "Search by company (case-insensitive substring match)"}
            },
            "required": []
        }
    },
    {
        "name": "search_qa_units",
        "description": "Search Q&A exchanges from the earnings call by speaker name, keyword in text, or follow-up status.",
        "parameters": {
            "type": "object",
            "properties": {
                "questioner_name": {"type": "string", "description": "Filter by questioner name (substring match)"},
                "responder_name": {"type": "string", "description": "Filter by responder name (substring match)"},
                "keyword": {"type": "string", "description": "Search keyword in question and response text (case-insensitive)"},
                "is_follow_up": {"type": "boolean", "description": "Filter for follow-up questions only"},
                "limit": {"type": "integer", "description": "Max results to return (default 5)", "default": 5}
            },
            "required": []
        }
    },
    {
        "name": "get_qa_detail",
        "description": "Get full details of a specific Q&A exchange by its ID (e.g., 'qa_001'). Use after search_qa_units to get complete text.",
        "parameters": {
            "type": "object",
            "properties": {
                "qa_id": {"type": "string", "description": "The Q&A unit ID (e.g., 'qa_000', 'qa_001')"}
            },
            "required": ["qa_id"]
        }
    },
    {
        "name": "get_follow_up_chain",
        "description": "Get the full chain of follow-up Q&A exchanges starting from a given Q&A ID. Returns the original question and all follow-ups in order.",
        "parameters": {
            "type": "object",
            "properties": {
                "qa_id": {"type": "string", "description": "Starting Q&A ID to trace the follow-up chain from"}
            },
            "required": ["qa_id"]
        }
    },
    {
        "name": "search_full_text",
        "description": "Search the raw transcript text for a keyword. Returns matching excerpts with page numbers. Use when structured search returns no results.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Keyword to search for in raw transcript text"},
                "context_chars": {"type": "integer", "description": "Characters of context around match (default 300)", "default": 300}
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "search_strategic_statements",
        "description": "Search strategic statements (guidance, outlook, initiatives) from the earnings call.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Search keyword in statement text"},
                "statement_type": {"type": "string", "enum": ["guidance", "outlook", "strategic_initiative", "operational_update", "financial_highlight", "risk_disclosure"]},
                "is_forward_looking": {"type": "boolean", "description": "Filter for forward-looking statements"},
                "speaker_name": {"type": "string", "description": "Filter by speaker name"}
            },
            "required": []
        }
    },
    {
        "name": "get_raw_text_page",
        "description": "Get the raw transcript text for a specific page number.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_number": {"type": "integer", "description": "Page number (1-based)"}
            },
            "required": ["page_number"]
        }
    }
]


# ============================================================
# Tool Execution (simulate real tool behavior)
# ============================================================

def execute_tool(tool_name: str, args: dict, run_data: dict) -> str:
    """Execute a tool against run data and return the result as a string."""

    if tool_name == "get_run_metadata":
        meta = run_data.get("meta", {})
        metadata = run_data.get("metadata", {})
        qa = run_data.get("qa", {})
        speakers = run_data.get("speakers", {})
        return json.dumps({
            "company_name": meta.get("company_name"),
            "ticker_symbol": meta.get("ticker_symbol"),
            "fiscal_quarter": meta.get("fiscal_quarter"),
            "fiscal_year": meta.get("fiscal_year"),
            "call_date": meta.get("call_date"),
            "speaker_count": len(speakers.get("speakers", {})),
            "qa_count": len(qa.get("qa_units", [])),
            "page_count": metadata.get("page_count", "?"),
        }, indent=2)

    elif tool_name == "search_speakers":
        speakers = run_data.get("speakers", {}).get("speakers", {})
        results = []
        for s in speakers.values():
            if args.get("role") and s.get("role") != args["role"]:
                continue
            if args.get("name_query"):
                q = args["name_query"].lower()
                name_match = q in s.get("canonical_name", "").lower()
                alias_match = any(q in a.lower() for a in s.get("aliases", []))
                if not name_match and not alias_match:
                    continue
            if args.get("company"):
                if not s.get("company") or args["company"].lower() not in s["company"].lower():
                    continue
            results.append({
                "speaker_id": s["speaker_id"],
                "canonical_name": s["canonical_name"],
                "role": s.get("role"),
                "title": s.get("title"),
                "company": s.get("company"),
                "turn_count": s.get("turn_count"),
            })
        return json.dumps({"results": results[:20], "total": len(results)}, indent=2)

    elif tool_name == "search_qa_units":
        qa_units = run_data.get("qa", {}).get("qa_units", [])
        results = []
        for q in qa_units:
            if args.get("questioner_name"):
                if args["questioner_name"].lower() not in q.get("questioner_name", "").lower():
                    continue
            if args.get("responder_name"):
                if not any(args["responder_name"].lower() in r.lower() for r in q.get("responder_names", [])):
                    continue
            if args.get("keyword"):
                kw = args["keyword"].lower()
                text = (q.get("question_text", "") + " " + q.get("response_text", "")).lower()
                if kw not in text:
                    continue
            if args.get("is_follow_up") is not None:
                if q.get("is_follow_up") != args["is_follow_up"]:
                    continue

            limit = args.get("limit", 5)
            # Adaptive truncation
            budget = 3000
            chars_per = budget // max(limit, 1)

            results.append({
                "qa_id": q["qa_id"],
                "questioner_name": q.get("questioner_name"),
                "responder_names": q.get("responder_names", []),
                "question_text": q.get("question_text", "")[:chars_per],
                "response_text": q.get("response_text", "")[:chars_per],
                "is_follow_up": q.get("is_follow_up"),
                "follow_up_of": q.get("follow_up_of"),
                "start_page": q.get("start_page"),
                "end_page": q.get("end_page"),
            })
            if len(results) >= limit:
                break

        return json.dumps({"results": results, "total_matching": len(results)}, indent=2)

    elif tool_name == "get_qa_detail":
        qa_units = run_data.get("qa", {}).get("qa_units", [])
        qa_id = args.get("qa_id", "")
        for q in qa_units:
            if q["qa_id"] == qa_id:
                return json.dumps(q, indent=2)
        return json.dumps({"error": f"Q&A unit '{qa_id}' not found"})

    elif tool_name == "get_follow_up_chain":
        qa_units = run_data.get("qa", {}).get("qa_units", [])
        qa_map = {q["qa_id"]: q for q in qa_units}
        qa_id = args.get("qa_id", "")

        # Find root
        current = qa_map.get(qa_id)
        if not current:
            return json.dumps({"error": f"Q&A unit '{qa_id}' not found"})

        while current.get("follow_up_of") and current["follow_up_of"] in qa_map:
            current = qa_map[current["follow_up_of"]]

        # Collect chain
        chain = [current]
        visited = {current["qa_id"]}
        # Find all that follow_up_of points to items in chain
        changed = True
        while changed:
            changed = False
            for q in qa_units:
                if q["qa_id"] not in visited and q.get("follow_up_of") in visited:
                    chain.append(q)
                    visited.add(q["qa_id"])
                    changed = True

        chain.sort(key=lambda x: x.get("sequence_in_session", 0))
        return json.dumps({"chain": [{"qa_id": q["qa_id"], "questioner": q.get("questioner_name"), "question_text": q.get("question_text", "")[:500], "response_text": q.get("response_text", "")[:500]} for q in chain]}, indent=2)

    elif tool_name == "search_full_text":
        extraction = run_data.get("extraction", {})
        keyword = args.get("keyword", "").lower()
        context_chars = args.get("context_chars", 300)

        results = []
        # Check both possible structures
        pages = extraction.get("pages", [])
        if not pages and "raw_text" in extraction:
            # Single text blob - split by page markers if possible
            pages = [{"page_number": 1, "text": extraction["raw_text"]}]

        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page_number", 0)
            idx = text.lower().find(keyword)
            while idx != -1:
                start = max(0, idx - context_chars // 2)
                end = min(len(text), idx + len(keyword) + context_chars // 2)
                results.append({
                    "page_number": page_num,
                    "excerpt": text[start:end],
                    "match_position": idx
                })
                idx = text.lower().find(keyword, idx + 1)

        return json.dumps({"results": results[:5], "total_matches": len(results)}, indent=2)

    elif tool_name == "search_strategic_statements":
        strategic = run_data.get("strategic", {})
        statements = strategic.get("statements", [])
        results = []
        for s in statements:
            if args.get("keyword") and args["keyword"].lower() not in s.get("text", "").lower():
                continue
            if args.get("statement_type") and s.get("statement_type") != args["statement_type"]:
                continue
            if args.get("is_forward_looking") is not None and s.get("is_forward_looking") != args["is_forward_looking"]:
                continue
            if args.get("speaker_name") and args["speaker_name"].lower() not in s.get("speaker_name", "").lower():
                continue
            results.append(s)
        return json.dumps({"results": results[:10]}, indent=2)

    elif tool_name == "get_raw_text_page":
        extraction = run_data.get("extraction", {})
        page_num = args.get("page_number", 1)
        pages = extraction.get("pages", [])
        for page in pages:
            if page.get("page_number") == page_num:
                return json.dumps({"page_number": page_num, "text": page.get("text", "")[:3000]})
        return json.dumps({"error": f"Page {page_num} not found"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ============================================================
# Test Result Tracking
# ============================================================

@dataclass
class TestResult:
    test_name: str
    approach: str  # "native" or "prompt_based"
    question: str
    success: bool = False
    tool_calls: list = field(default_factory=list)
    expected_tools: list = field(default_factory=list)
    tool_selection_correct: bool = False
    params_correct: bool = False
    response_text: str = ""
    response_quality: str = ""  # "good", "partial", "bad", "hallucinated"
    time_per_call_seconds: list = field(default_factory=list)
    total_time_seconds: float = 0.0
    error: str = ""
    raw_output: str = ""


# ============================================================
# Test: Native Tool Calling (ChatOllama)
# ============================================================

def test_native_tool_calling(model_name: str, num_ctx: int, run_data: dict, data_summary: str) -> list[TestResult]:
    """Test native ChatOllama tool calling."""
    print("\n" + "=" * 70)
    print("TEST SUITE: Native Tool Calling (ChatOllama)")
    print("=" * 70)

    results = []

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
        from langchain_core.tools import tool as tool_decorator

        llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=num_ctx,
        )

        # Convert tool definitions to langchain tools
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField, create_model

        # Build pydantic models for tools dynamically
        langchain_tools = []
        for td in TOOL_DEFINITIONS:
            props = td["parameters"].get("properties", {})
            fields = {}
            for pname, pdef in props.items():
                ptype = str
                if pdef.get("type") == "integer":
                    ptype = int
                elif pdef.get("type") == "boolean":
                    ptype = bool

                if pname in td["parameters"].get("required", []):
                    fields[pname] = (ptype, PydanticField(description=pdef.get("description", "")))
                else:
                    fields[pname] = (Optional[ptype], PydanticField(default=None, description=pdef.get("description", "")))

            if fields:
                args_model = create_model(f"{td['name']}_args", **fields)
            else:
                args_model = create_model(f"{td['name']}_args")

            def make_func(name):
                def func(**kwargs):
                    return execute_tool(name, kwargs, run_data)
                func.__name__ = name
                func.__doc__ = td["description"]
                return func

            st = StructuredTool.from_function(
                func=make_func(td["name"]),
                name=td["name"],
                description=td["description"],
                args_schema=args_model,
            )
            langchain_tools.append(st)

        llm_with_tools = llm.bind_tools(langchain_tools)

    except Exception as e:
        print(f"\n  FATAL: Failed to initialize ChatOllama with tools: {e}")
        result = TestResult(
            test_name="initialization",
            approach="native",
            question="N/A",
            error=str(e),
        )
        return [result]

    # System prompt
    system_prompt = f"""You are an earnings call analyst assistant. You answer questions about an earnings call by using the available tools.

DATA SUMMARY:
{data_summary}

RULES:
1. Always use tools to find data. Never fabricate information.
2. If search returns no results, try search_full_text before concluding the topic wasn't discussed.
3. When referencing transcript content, cite using [qa_XXX], [page_N] format.
4. Keep answers concise."""

    test_scenarios = [
        {
            "name": "simple_factual",
            "question": "Who are the management speakers in this call?",
            "expected_tools": ["search_speakers"],
            "validate": lambda calls: any("search_speakers" in str(c) for c in calls),
        },
        {
            "name": "keyword_search",
            "question": "What was discussed about EBITDA margins?",
            "expected_tools": ["search_qa_units"],
            "validate": lambda calls: any("search_qa_units" in str(c) for c in calls),
        },
        {
            "name": "metadata_lookup",
            "question": "What company and quarter is this call for?",
            "expected_tools": ["get_run_metadata"],
            "validate": lambda calls: any("get_run_metadata" in str(c) for c in calls),
        },
        {
            "name": "full_text_fallback",
            "question": "Was there any discussion about zinc oxide applications?",
            "expected_tools": ["search_qa_units", "search_full_text"],
            "validate": lambda calls: any("search_qa_units" in str(c) or "search_full_text" in str(c) for c in calls),
        },
        {
            "name": "multi_step",
            "question": "Give me a summary of the main topics discussed in the Q&A session.",
            "expected_tools": ["search_qa_units", "get_run_metadata"],
            "validate": lambda calls: len(calls) >= 1,
        },
    ]

    for scenario in test_scenarios:
        print(f"\n  Test: {scenario['name']}")
        print(f"  Question: {scenario['question']}")

        result = TestResult(
            test_name=scenario["name"],
            approach="native",
            question=scenario["question"],
            expected_tools=scenario["expected_tools"],
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=scenario["question"]),
            ]

            all_tool_calls = []
            call_times = []
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                start = time.time()
                response = llm_with_tools.invoke(messages)
                elapsed = time.time() - start
                call_times.append(elapsed)

                print(f"    Iteration {iteration}: {elapsed:.1f}s")

                if not response.tool_calls:
                    # Final response
                    result.response_text = response.content
                    print(f"    Final response: {response.content[:200]}...")
                    break

                # Process tool calls
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    all_tool_calls.append({"name": tool_name, "args": tool_args})
                    print(f"    Tool call: {tool_name}({json.dumps(tool_args)})")

                    # Execute tool
                    tool_result = execute_tool(tool_name, tool_args, run_data)
                    messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))

            result.tool_calls = all_tool_calls
            result.time_per_call_seconds = call_times
            result.total_time_seconds = sum(call_times)
            result.success = True
            result.tool_selection_correct = scenario["validate"](all_tool_calls)

            # Basic quality check
            if result.response_text:
                result.response_quality = "good" if len(result.response_text) > 50 else "partial"
            else:
                result.response_quality = "no_response"

            print(f"    Tool selection correct: {result.tool_selection_correct}")
            print(f"    Total time: {result.total_time_seconds:.1f}s")

        except Exception as e:
            result.error = str(e)
            result.response_quality = "error"
            print(f"    ERROR: {e}")

        results.append(result)

    return results


# ============================================================
# Test: Prompt-Based Tool Calling (OllamaLLM)
# ============================================================

def test_prompt_based_tool_calling(model_name: str, num_ctx: int, run_data: dict, data_summary: str) -> list[TestResult]:
    """Test prompt-based tool calling using OllamaLLM with text parsing."""
    print("\n" + "=" * 70)
    print("TEST SUITE: Prompt-Based Tool Calling (OllamaLLM)")
    print("=" * 70)

    results = []

    try:
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=num_ctx,
            num_predict=4096,
            streaming=False,
        )
    except Exception as e:
        print(f"\n  FATAL: Failed to initialize OllamaLLM: {e}")
        return [TestResult(test_name="initialization", approach="prompt_based", question="N/A", error=str(e))]

    # Build tool descriptions for the prompt
    tool_descriptions = ""
    for td in TOOL_DEFINITIONS:
        params = td["parameters"].get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'string')}" + (f" (required)" if k in td["parameters"].get("required", []) else " (optional)")
            for k, v in params.items()
        )
        tool_descriptions += f"\n- {td['name']}({param_str}): {td['description']}"

    system_prompt = f"""You are an earnings call analyst assistant. You answer questions by calling tools and synthesizing results.

DATA SUMMARY:
{data_summary}

AVAILABLE TOOLS:
{tool_descriptions}

HOW TO CALL TOOLS:
When you need data, output a tool call in this exact format:
TOOL_CALL: tool_name(param1="value1", param2="value2")

After receiving tool results, you can call more tools or provide your final answer.
When you have enough information, provide your answer directly (without TOOL_CALL).

RULES:
1. Always use tools to find data. Never fabricate information.
2. If a search returns no results, try search_full_text before concluding the topic wasn't discussed.
3. Reference Q&A units by ID [qa_XXX] and pages [page_N].
4. Keep answers concise."""

    def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
        """Parse a TOOL_CALL from LLM output."""
        # Match TOOL_CALL: function_name(params)
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        func_name = match.group(1)
        params_str = match.group(2).strip()

        # Parse parameters
        args = {}
        if params_str:
            # Try to parse key="value" pairs
            param_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+)|(\w+))'
            for pm in re.finditer(param_pattern, params_str):
                key = pm.group(1)
                value = pm.group(2) or pm.group(3) or pm.group(4) or pm.group(5)
                # Convert types
                if value and value.isdigit():
                    value = int(value)
                elif value in ("true", "True"):
                    value = True
                elif value in ("false", "False"):
                    value = False
                args[key] = value

        return func_name, args

    test_scenarios = [
        {
            "name": "simple_factual",
            "question": "Who are the management speakers in this call?",
            "expected_tools": ["search_speakers"],
            "validate": lambda calls: any("search_speakers" in c.get("name", "") for c in calls),
        },
        {
            "name": "keyword_search",
            "question": "What was discussed about EBITDA margins?",
            "expected_tools": ["search_qa_units"],
            "validate": lambda calls: any("search_qa_units" in c.get("name", "") for c in calls),
        },
        {
            "name": "metadata_lookup",
            "question": "What company and quarter is this call for?",
            "expected_tools": ["get_run_metadata"],
            "validate": lambda calls: any("get_run_metadata" in c.get("name", "") for c in calls),
        },
        {
            "name": "full_text_fallback",
            "question": "Was there any discussion about zinc oxide applications?",
            "expected_tools": ["search_qa_units", "search_full_text"],
            "validate": lambda calls: any(c.get("name", "") in ("search_qa_units", "search_full_text") for c in calls),
        },
        {
            "name": "multi_step",
            "question": "Give me a summary of the main topics discussed in the Q&A session.",
            "expected_tools": ["search_qa_units", "get_run_metadata"],
            "validate": lambda calls: len(calls) >= 1,
        },
    ]

    for scenario in test_scenarios:
        print(f"\n  Test: {scenario['name']}")
        print(f"  Question: {scenario['question']}")

        result = TestResult(
            test_name=scenario["name"],
            approach="prompt_based",
            question=scenario["question"],
            expected_tools=scenario["expected_tools"],
        )

        try:
            conversation = f"{system_prompt}\n\nUser: {scenario['question']}\nAssistant:"

            all_tool_calls = []
            call_times = []
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                start = time.time()
                response_text = llm.invoke(conversation)
                elapsed = time.time() - start
                call_times.append(elapsed)

                print(f"    Iteration {iteration}: {elapsed:.1f}s")

                # Check for tool call
                parsed = parse_tool_call(response_text)
                if parsed:
                    tool_name, tool_args = parsed
                    all_tool_calls.append({"name": tool_name, "args": tool_args})
                    print(f"    Tool call: {tool_name}({json.dumps(tool_args)})")

                    # Execute and append to conversation
                    tool_result = execute_tool(tool_name, tool_args, run_data)
                    conversation += f" {response_text}\n\nTool Result:\n{tool_result}\n\nAssistant:"
                else:
                    # Final response
                    result.response_text = response_text.strip()
                    result.raw_output = response_text
                    # Show what the model actually returned
                    safe_preview = response_text[:400].encode('ascii', errors='replace').decode('ascii')
                    print(f"    Final response ({len(response_text)} chars):")
                    print(f"    >>> {safe_preview}")
                    break

            result.tool_calls = all_tool_calls
            result.time_per_call_seconds = call_times
            result.total_time_seconds = sum(call_times)
            result.success = True
            result.tool_selection_correct = scenario["validate"](all_tool_calls)

            if result.response_text:
                result.response_quality = "good" if len(result.response_text) > 50 else "partial"
            elif all_tool_calls:
                result.response_quality = "tools_only_no_synthesis"
            else:
                result.response_quality = "no_response"

            print(f"    Tool selection correct: {result.tool_selection_correct}")
            print(f"    Total time: {result.total_time_seconds:.1f}s")

        except Exception as e:
            result.error = str(e)
            result.response_quality = "error"
            print(f"    ERROR: {e}")

        results.append(result)

    return results


# ============================================================
# Report Generation
# ============================================================

def generate_report(native_results: list[TestResult], prompt_results: list[TestResult], model_name: str, num_ctx: int):
    """Generate a detailed comparison report."""
    print("\n\n" + "=" * 70)
    print("DETAILED ANALYSIS REPORT")
    print(f"Model: {model_name} | Context: {num_ctx} tokens")
    print("=" * 70)

    for approach_name, results in [("Native Tool Calling (ChatOllama)", native_results), ("Prompt-Based Tool Calling (OllamaLLM)", prompt_results)]:
        print(f"\n{'-' * 50}")
        print(f"  {approach_name}")
        print(f"{'-' * 50}")

        if not results:
            print("  No results (initialization failed)")
            continue

        # Check for init failure
        if results[0].test_name == "initialization" and results[0].error:
            print(f"  INITIALIZATION FAILED: {results[0].error}")
            continue

        total_tests = len(results)
        successful = sum(1 for r in results if r.success)
        correct_tools = sum(1 for r in results if r.tool_selection_correct)

        all_times = []
        for r in results:
            all_times.extend(r.time_per_call_seconds)

        avg_time = sum(all_times) / len(all_times) if all_times else 0
        total_times = [r.total_time_seconds for r in results if r.total_time_seconds > 0]
        avg_total = sum(total_times) / len(total_times) if total_times else 0

        print(f"\n  Summary:")
        print(f"    Tests passed:           {successful}/{total_tests}")
        print(f"    Correct tool selection: {correct_tools}/{total_tests}")
        print(f"    Avg time per LLM call:  {avg_time:.1f}s")
        print(f"    Avg total response time: {avg_total:.1f}s")
        if all_times:
            print(f"    Min/Max call time:      {min(all_times):.1f}s / {max(all_times):.1f}s")

        print(f"\n  Per-Test Details:")
        for r in results:
            if r.test_name == "initialization":
                continue
            status = "PASS" if r.success and r.tool_selection_correct else "FAIL" if r.error else "PARTIAL"
            print(f"\n    [{status}] {r.test_name}")
            print(f"      Question: {r.question}")
            print(f"      Expected tools: {r.expected_tools}")
            print(f"      Actual tools:   {[c['name'] for c in r.tool_calls]}")
            print(f"      Tool params:    {[c.get('args', {}) for c in r.tool_calls]}")
            print(f"      Correct tools:  {r.tool_selection_correct}")
            print(f"      Response quality: {r.response_quality}")
            print(f"      Time: {r.total_time_seconds:.1f}s ({len(r.time_per_call_seconds)} calls)")
            if r.error:
                print(f"      Error: {r.error}")
            if r.response_text:
                preview = r.response_text[:300].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
                print(f"      Response preview: {preview}...")

    # Comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON & RECOMMENDATION")
    print(f"{'=' * 70}")

    native_ok = any(r.success and r.tool_selection_correct for r in native_results if r.test_name != "initialization")
    prompt_ok = any(r.success and r.tool_selection_correct for r in prompt_results if r.test_name != "initialization")

    native_init_failed = any(r.test_name == "initialization" and r.error for r in native_results)

    if native_init_failed:
        print("\n  Native tool calling: FAILED TO INITIALIZE")
        print(f"  Error: {native_results[0].error}")
    else:
        native_correct = sum(1 for r in native_results if r.tool_selection_correct and r.test_name != "initialization")
        print(f"\n  Native tool calling: {native_correct}/{len([r for r in native_results if r.test_name != 'initialization'])} tests passed")

    prompt_correct = sum(1 for r in prompt_results if r.tool_selection_correct and r.test_name != "initialization")
    prompt_total = len([r for r in prompt_results if r.test_name != "initialization"])
    print(f"  Prompt-based calling: {prompt_correct}/{prompt_total} tests passed")

    # Timing comparison
    native_times = [r.total_time_seconds for r in native_results if r.total_time_seconds > 0]
    prompt_times = [r.total_time_seconds for r in prompt_results if r.total_time_seconds > 0]

    if native_times:
        print(f"\n  Native avg response time: {sum(native_times)/len(native_times):.1f}s")
    if prompt_times:
        print(f"  Prompt avg response time: {sum(prompt_times)/len(prompt_times):.1f}s")

    print(f"\n  RECOMMENDATION:")
    if native_init_failed and prompt_ok:
        print("  → Use PROMPT-BASED approach (native tool calling not supported)")
    elif native_ok and not prompt_ok:
        print("  → Use NATIVE tool calling (prompt-based parsing unreliable)")
    elif native_ok and prompt_ok:
        n_score = sum(1 for r in native_results if r.tool_selection_correct)
        p_score = sum(1 for r in prompt_results if r.tool_selection_correct)
        if n_score >= p_score:
            print("  → Use NATIVE tool calling (both work, native is cleaner)")
        else:
            print("  → Use PROMPT-BASED approach (more reliable tool selection)")
    elif not native_ok and not prompt_ok:
        print("  → NEITHER approach works reliably with this model")
        print("  → Consider: qwen2.5:14b, llama3.1:8b, or a cloud API")

    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test LLM tool calling for chatbot agent")
    parser.add_argument("--run-id", default="20e9f23c-744", help="Run ID to test against (partial match supported)")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Context window size")
    parser.add_argument("--skip-native", action="store_true", help="Skip native tool calling test")
    parser.add_argument("--skip-prompt", action="store_true", help="Skip prompt-based test")
    args = parser.parse_args()

    print(f"Loading run data: {args.run_id}")
    run_data = load_run_data(args.run_id)
    data_summary = generate_data_summary(run_data)

    print(f"\nData Summary:\n{data_summary}")
    print(f"\nModel: {args.model}")
    print(f"Context window: {args.num_ctx}")

    native_results = []
    prompt_results = []

    if not args.skip_native:
        native_results = test_native_tool_calling(args.model, args.num_ctx, run_data, data_summary)

    if not args.skip_prompt:
        prompt_results = test_prompt_based_tool_calling(args.model, args.num_ctx, run_data, data_summary)

    generate_report(native_results, prompt_results, args.model, args.num_ctx)


if __name__ == "__main__":
    main()
