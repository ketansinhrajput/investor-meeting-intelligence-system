"""Model comparison test for chat agent.

Tests local Ollama models on:
1. Phase 1: Tool selection (JSON output reliability)
2. Phase 2: Grounded synthesis (answer quality, citation adherence)
3. Summary synthesis (thematic grouping vs QA dumping)

Run: python scripts/test_model_comparison.py
"""

import json
import os
import sys
import time

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

from langchain_ollama import OllamaLLM

from services.chat_prompts import (
    build_analyst_synthesis_prompt,
    build_summary_synthesis_prompt,
    build_synthesis_prompt,
    build_tool_selection_prompt,
)

# ============================================================
# Config
# ============================================================

MODELS_TO_TEST = [
    "gpt-oss:20b",
    "gemma3n:e4b",
    "qwen3-vl:8b",
]

LLM_BASE_URL = "http://localhost:11434"
LLM_NUM_CTX = 16384
LLM_TEMPERATURE = 0.1
PHASE1_RETRIES = 2  # per question

# ============================================================
# Test Data (simulated evidence)
# ============================================================

MOCK_SUMMARY = {
    "company": "Acme Corp",
    "quarter": "Q3",
    "year": 2025,
    "speaker_count": 8,
    "qa_count": 15,
    "management_names": ["John Smith (CEO)", "Jane Doe (CFO)"],
    "analyst_names": ["Nitin Gandhi", "Sarah Lee", "Mike Chen"],
}

MOCK_EVIDENCE = """{
  "results": [
    {
      "qa_id": "qa_003",
      "questioner_name": "Nitin Gandhi",
      "responder_names": ["Jane Doe"],
      "question_text": "Can you talk about the margin trajectory going into next quarter? We have seen some compression in the last two quarters.",
      "response_text": "Sure Nitin. We expect margins to stabilize around 18-19% as input costs normalize. We have also renegotiated key supplier contracts which should help from Q4 onwards.",
      "start_page": 12,
      "is_follow_up": false
    },
    {
      "qa_id": "qa_007",
      "questioner_name": "Sarah Lee",
      "responder_names": ["John Smith"],
      "question_text": "What is the outlook for international expansion, particularly in Southeast Asia?",
      "response_text": "We are actively evaluating opportunities in Vietnam and Indonesia. We expect to announce a partnership in Q1 next year. The TAM in Southeast Asia is significant.",
      "start_page": 15,
      "is_follow_up": false
    },
    {
      "qa_id": "qa_011",
      "questioner_name": "Mike Chen",
      "responder_names": ["Jane Doe", "John Smith"],
      "question_text": "Can you provide color on the increase in R&D spending? Is this sustainable?",
      "response_text": "We have been investing heavily in our AI platform. R&D is up 22% YoY. We believe this positions us well for 2026 but will moderate spending as products launch.",
      "start_page": 18,
      "is_follow_up": false
    },
    {
      "qa_id": "qa_012",
      "questioner_name": "Nitin Gandhi",
      "responder_names": ["Jane Doe"],
      "question_text": "Just a follow-up on margins - is the supplier renegotiation already reflected in Q3 numbers?",
      "response_text": "No, the new contracts take effect in Q4. So the Q3 margin of 17.2% is pre-benefit. You should see the uplift starting next quarter.",
      "start_page": 19,
      "is_follow_up": true
    }
  ],
  "total_matching": 4,
  "showing": 4
}"""

# Summary evidence (all QAs for summary test)
MOCK_SUMMARY_EVIDENCE = """{
  "results": [
    {"qa_id": "qa_001", "questioner_name": "Sarah Lee", "question_text": "What drove the revenue beat this quarter?", "response_text": "Strong enterprise demand and new customer wins in North America."},
    {"qa_id": "qa_002", "questioner_name": "Mike Chen", "question_text": "How should we think about capex for next year?", "response_text": "We expect capex to remain elevated as we build out data centers."},
    {"qa_id": "qa_003", "questioner_name": "Nitin Gandhi", "question_text": "Can you talk about the margin trajectory?", "response_text": "Margins should stabilize at 18-19% as input costs normalize."},
    {"qa_id": "qa_004", "questioner_name": "Sarah Lee", "question_text": "Any update on the product pipeline?", "response_text": "We plan to launch three new products in Q1, focused on AI and automation."},
    {"qa_id": "qa_005", "questioner_name": "Nitin Gandhi", "question_text": "What about competitive pressures in the enterprise segment?", "response_text": "We see rational pricing. Our win rates have actually improved quarter over quarter."},
    {"qa_id": "qa_006", "questioner_name": "Mike Chen", "question_text": "Can you discuss the international expansion timeline?", "response_text": "Southeast Asia partnership expected Q1 next year. Europe is later in 2026."},
    {"qa_id": "qa_007", "questioner_name": "Sarah Lee", "question_text": "What is the outlook for international expansion?", "response_text": "Actively evaluating Vietnam and Indonesia. TAM in Southeast Asia is significant."},
    {"qa_id": "qa_008", "questioner_name": "Nitin Gandhi", "question_text": "Follow-up on margins - is supplier renegotiation in Q3?", "response_text": "No, new contracts in Q4. Q3 margin of 17.2% is pre-benefit."},
    {"qa_id": "qa_009", "questioner_name": "Mike Chen", "question_text": "R&D spending increase - is this sustainable?", "response_text": "R&D up 22% YoY for AI platform. Will moderate as products launch."},
    {"qa_id": "qa_010", "questioner_name": "Sarah Lee", "question_text": "What is the customer retention rate trend?", "response_text": "Net retention above 120% for enterprise, stable at 105% for SMB."}
  ],
  "total_matching": 10,
  "showing": 10
}"""

# ============================================================
# Test Questions
# ============================================================

PHASE1_QUESTIONS = [
    "What did Nitin Gandhi ask about?",
    "Tell me about margins",
    "Who are the management speakers?",
    "What topics were discussed in the Q&A?",
    "What is the company's guidance for next quarter?",
]

PHASE2_QUESTIONS = [
    ("What did analysts ask about margins?", MOCK_EVIDENCE),
    ("What is the international expansion outlook?", MOCK_EVIDENCE),
]

SUMMARY_QUESTION = "Summarize the key themes from the Q&A session"

ANALYST_QUESTION = ("What did Nitin Gandhi ask about?", MOCK_EVIDENCE)


# ============================================================
# Helpers
# ============================================================

def create_llm(model: str) -> OllamaLLM:
    return OllamaLLM(
        model=model,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        num_predict=1024,
        streaming=False,
    )


def parse_json_from_text(text: str):
    text = text.strip()
    if not text or "{" not in text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def score_phase1(response_text: str) -> dict:
    """Score a Phase 1 response for JSON validity and tool selection quality."""
    if not response_text or not response_text.strip():
        return {"valid_json": False, "has_action": False, "has_tool": False, "empty": True, "score": 0}

    parsed = parse_json_from_text(response_text)
    if not parsed:
        return {"valid_json": False, "has_action": False, "has_tool": False, "empty": False, "score": 1}

    has_action = "action" in parsed
    has_tool = "tool" in parsed
    has_params = "params" in parsed

    score = 2  # valid JSON
    if has_action:
        score += 2
    if has_tool:
        score += 3
    if has_params:
        score += 1
    # Bonus for known tool names
    known_tools = ["search_qa_units", "search_speakers", "get_run_metadata",
                   "search_full_text", "search_strategic_statements", "get_follow_up_chain"]
    if parsed.get("tool") in known_tools:
        score += 2

    return {
        "valid_json": True,
        "has_action": has_action,
        "has_tool": has_tool,
        "has_params": has_params,
        "tool": parsed.get("tool"),
        "empty": False,
        "score": score,
    }


def score_phase2(response_text: str) -> dict:
    """Score a Phase 2 response for quality indicators."""
    if not response_text or not response_text.strip():
        return {"empty": True, "score": 0, "citations": 0, "length": 0}

    import re
    text = response_text.strip()
    citations = len(re.findall(r"\[qa_\d+\]", text))
    fullwidth_citations = len(re.findall(r"[\u3010]qa_\d+[\u3011]", text))
    has_markdown = "**" in text or "- " in text or "* " in text
    has_insufficient = "insufficient" in text.lower() or "not contain" in text.lower()
    length = len(text)

    score = 0
    if length > 50:
        score += 2
    if length > 200:
        score += 1
    if citations > 0:
        score += min(citations * 2, 6)  # up to 6 pts for citations
    if fullwidth_citations > 0:
        score -= 1  # penalty for fullwidth brackets
    if has_markdown:
        score += 2
    if has_insufficient and citations == 0:
        score += 1  # at least it's honest
    # Penalize if it looks like raw QA dump
    qa_dump_indicators = text.count("Question:") + text.count("question_text") + text.count("response_text")
    if qa_dump_indicators > 2:
        score -= 3  # penalty for dumping raw QA

    return {
        "empty": False,
        "score": score,
        "citations": citations + fullwidth_citations,
        "fullwidth": fullwidth_citations,
        "has_markdown": has_markdown,
        "length": length,
        "qa_dump_penalty": qa_dump_indicators > 2,
    }


def safe_print(text: str):
    """Print with ASCII fallback for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ============================================================
# Main Test Runner
# ============================================================

def run_tests():
    results = {}

    for model_name in MODELS_TO_TEST:
        safe_print(f"\n{'='*70}")
        safe_print(f"TESTING: {model_name}")
        safe_print(f"{'='*70}")

        try:
            llm = create_llm(model_name)
            # Quick warmup
            _ = llm.invoke("Say OK")
        except Exception as e:
            safe_print(f"  SKIP - Cannot load model: {e}")
            results[model_name] = {"error": str(e)}
            continue

        model_results = {
            "phase1_scores": [],
            "phase2_scores": [],
            "summary_score": None,
            "analyst_score": None,
            "phase1_times": [],
            "phase2_times": [],
        }

        # --- Phase 1: Tool Selection ---
        safe_print(f"\n  PHASE 1: Tool Selection ({len(PHASE1_QUESTIONS)} questions, {PHASE1_RETRIES} attempts each)")
        safe_print(f"  {'-'*60}")

        speaker_names = ", ".join(MOCK_SUMMARY["management_names"] + MOCK_SUMMARY["analyst_names"])

        for q in PHASE1_QUESTIONS:
            best_score = None
            best_response = ""

            for attempt in range(PHASE1_RETRIES):
                prompt = build_tool_selection_prompt(
                    company=MOCK_SUMMARY["company"],
                    quarter=MOCK_SUMMARY["quarter"],
                    year=MOCK_SUMMARY["year"],
                    speaker_count=MOCK_SUMMARY["speaker_count"],
                    qa_count=MOCK_SUMMARY["qa_count"],
                    speaker_names=speaker_names,
                    user_question=q,
                )
                t0 = time.time()
                try:
                    resp = llm.invoke(prompt).strip()
                except Exception as e:
                    resp = ""
                    safe_print(f"    ERROR: {e}")
                elapsed = time.time() - t0
                model_results["phase1_times"].append(elapsed)

                sc = score_phase1(resp)
                if best_score is None or sc["score"] > best_score["score"]:
                    best_score = sc
                    best_response = resp

            model_results["phase1_scores"].append(best_score)
            status = "OK" if best_score["score"] >= 8 else "WEAK" if best_score["score"] >= 4 else "FAIL"
            tool_info = best_score.get("tool", "?")
            safe_print(f"    [{status}] score={best_score['score']:>2}  tool={tool_info:<25}  Q: {q[:50]}")
            if best_score["score"] < 4:
                safe_print(f"           Raw: {best_response[:120]}")

        # --- Phase 2: Synthesis ---
        safe_print(f"\n  PHASE 2: Grounded Synthesis ({len(PHASE2_QUESTIONS)} questions)")
        safe_print(f"  {'-'*60}")

        for q, evidence in PHASE2_QUESTIONS:
            prompt = build_synthesis_prompt(
                company=MOCK_SUMMARY["company"],
                quarter=MOCK_SUMMARY["quarter"],
                year=MOCK_SUMMARY["year"],
                user_question=q,
                evidence=evidence,
            )
            t0 = time.time()
            try:
                resp = llm.invoke(prompt).strip()
            except Exception as e:
                resp = ""
                safe_print(f"    ERROR: {e}")
            elapsed = time.time() - t0
            model_results["phase2_times"].append(elapsed)

            sc = score_phase2(resp)
            model_results["phase2_scores"].append(sc)
            status = "OK" if sc["score"] >= 6 else "WEAK" if sc["score"] >= 3 else "FAIL"
            safe_print(f"    [{status}] score={sc['score']:>2}  cites={sc['citations']}  len={sc['length']}  md={sc['has_markdown']}  Q: {q[:50]}")
            if sc["score"] < 3:
                safe_print(f"           Raw: {resp[:200]}")
            elif sc.get("qa_dump_penalty"):
                safe_print(f"           WARNING: Looks like raw QA dump")

        # --- Summary Test ---
        safe_print(f"\n  SUMMARY SYNTHESIS TEST")
        safe_print(f"  {'-'*60}")

        prompt = build_summary_synthesis_prompt(
            company=MOCK_SUMMARY["company"],
            quarter=MOCK_SUMMARY["quarter"],
            year=MOCK_SUMMARY["year"],
            user_question=SUMMARY_QUESTION,
            evidence=MOCK_SUMMARY_EVIDENCE,
        )
        t0 = time.time()
        try:
            resp = llm.invoke(prompt).strip()
        except Exception as e:
            resp = ""
            safe_print(f"    ERROR: {e}")
        elapsed = time.time() - t0

        sc = score_phase2(resp)
        model_results["summary_score"] = sc
        model_results["phase2_times"].append(elapsed)
        status = "OK" if sc["score"] >= 6 else "WEAK" if sc["score"] >= 3 else "FAIL"
        safe_print(f"    [{status}] score={sc['score']:>2}  cites={sc['citations']}  len={sc['length']}  dump={sc.get('qa_dump_penalty',False)}")
        safe_print(f"    Preview: {resp[:300]}...")

        # --- Analyst-Specific Test ---
        safe_print(f"\n  ANALYST QUESTION TEST (Nitin Gandhi)")
        safe_print(f"  {'-'*60}")

        q, evidence = ANALYST_QUESTION
        prompt = build_analyst_synthesis_prompt(
            company=MOCK_SUMMARY["company"],
            quarter=MOCK_SUMMARY["quarter"],
            year=MOCK_SUMMARY["year"],
            analyst_name="Nitin Gandhi",
            user_question=q,
            evidence=evidence,
        )
        t0 = time.time()
        try:
            resp = llm.invoke(prompt).strip()
        except Exception as e:
            resp = ""
            safe_print(f"    ERROR: {e}")
        elapsed = time.time() - t0

        sc = score_phase2(resp)
        model_results["analyst_score"] = sc
        model_results["phase2_times"].append(elapsed)
        status = "OK" if sc["score"] >= 6 else "WEAK" if sc["score"] >= 3 else "FAIL"
        safe_print(f"    [{status}] score={sc['score']:>2}  cites={sc['citations']}  len={sc['length']}")
        safe_print(f"    Preview: {resp[:300]}...")

        results[model_name] = model_results

    # ============================================================
    # Final Comparison
    # ============================================================
    safe_print(f"\n\n{'='*70}")
    safe_print("FINAL COMPARISON")
    safe_print(f"{'='*70}\n")

    safe_print(f"{'Model':<25} {'P1 Avg':>8} {'P1 JSON%':>9} {'P2 Avg':>8} {'Summary':>8} {'Analyst':>8} {'Avg Time':>9}")
    safe_print("-" * 80)

    for model_name, r in results.items():
        if "error" in r:
            safe_print(f"{model_name:<25} {'ERROR':>8}")
            continue

        p1_scores = [s["score"] for s in r["phase1_scores"]]
        p1_avg = sum(p1_scores) / len(p1_scores) if p1_scores else 0
        p1_json_pct = sum(1 for s in r["phase1_scores"] if s.get("valid_json")) / len(r["phase1_scores"]) * 100 if r["phase1_scores"] else 0

        p2_scores = [s["score"] for s in r["phase2_scores"]]
        p2_avg = sum(p2_scores) / len(p2_scores) if p2_scores else 0

        sum_score = r["summary_score"]["score"] if r["summary_score"] else 0
        analyst_score = r["analyst_score"]["score"] if r["analyst_score"] else 0

        all_times = r["phase1_times"] + r["phase2_times"]
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        safe_print(f"{model_name:<25} {p1_avg:>8.1f} {p1_json_pct:>8.0f}% {p2_avg:>8.1f} {sum_score:>8} {analyst_score:>8} {avg_time:>8.1f}s")

    safe_print("\nScoring: P1 max=10 (valid JSON + correct tool), P2 max=11 (citations + markdown + length)")
    safe_print("Higher is better for all metrics. Lower time is better.")


if __name__ == "__main__":
    run_tests()
