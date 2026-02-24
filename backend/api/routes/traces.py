"""
Traces Route

Endpoints for retrieving LLM decision traces and debugging information.

This is critical for understanding WHY the pipeline made specific decisions.
Each trace shows:
- What context the LLM received
- What decision it made
- What evidence supported the decision
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    TracesResponse,
    StageTrace,
    TraceDecision,
    EvidenceSpanResponse,
)
from services import storage

router = APIRouter()


def _extract_decisions_from_speakers_trace(trace: dict) -> list[TraceDecision]:
    """Extract decisions from speaker registry trace."""
    decisions = []

    # Verification decisions
    for i, decision in enumerate(trace.get("verification_decisions", [])):
        evidence_spans = []
        for span in decision.get("evidence_spans", []):
            if isinstance(span, dict):
                evidence_spans.append(EvidenceSpanResponse(
                    text=span.get("text", "")[:300],
                    source=span.get("source", "unknown"),
                    relevance=span.get("relevance", ""),
                ))

        decisions.append(TraceDecision(
            decision_id=f"speaker_verification_{i}",
            decision_type="speaker_verification",
            input_context=f"Candidate: {decision.get('candidate_name', 'Unknown')}",
            output_decision=f"Real person: {decision.get('is_real_person', False)}, "
                          f"Role: {decision.get('role', 'unknown')}, "
                          f"Canonical: {decision.get('canonical_name', '')}",
            confidence=decision.get("role_confidence"),
            reasoning=decision.get("reasoning", ""),
            evidence_spans=evidence_spans,
        ))

    # Alias merge decisions
    for i, decision in enumerate(trace.get("alias_merge_decisions", [])):
        evidence_spans = []
        for span in decision.get("evidence_spans", []):
            if isinstance(span, dict):
                evidence_spans.append(EvidenceSpanResponse(
                    text=span.get("text", "")[:300],
                    source=span.get("source", "unknown"),
                    relevance=span.get("relevance", ""),
                ))

        decisions.append(TraceDecision(
            decision_id=f"alias_merge_{i}",
            decision_type="alias_merge",
            input_context=f"Name1: {decision.get('name1', '')}, Name2: {decision.get('name2', '')}",
            output_decision=f"Merged: {decision.get('merged', False)}",
            confidence=decision.get("llm_confidence"),
            reasoning=decision.get("reason", ""),
            evidence_spans=evidence_spans,
        ))

    # Title assignments
    for i, decision in enumerate(trace.get("title_assignments", [])):
        decisions.append(TraceDecision(
            decision_id=f"title_assignment_{i}",
            decision_type="title_assignment",
            input_context=f"Speaker: {decision.get('speaker_name', '')}",
            output_decision=f"Title: {decision.get('title', 'None')}, "
                          f"Verified: {decision.get('title_verified', False)}",
            reasoning=decision.get("rejection_reason", "") or decision.get("evidence_text", "")[:200],
            evidence_spans=[],
        ))

    return decisions


def _extract_decisions_from_qa_trace(trace: dict) -> list[TraceDecision]:
    """Extract decisions from Q&A extraction trace."""
    decisions = []

    # Turn intent records
    for i, record in enumerate(trace.get("turn_intent_records", [])):
        evidence_spans = []
        for span in record.get("evidence_spans", []):
            if isinstance(span, dict):
                evidence_spans.append(EvidenceSpanResponse(
                    text=span.get("text", "")[:300],
                    source=span.get("source", "unknown"),
                    relevance=span.get("relevance", ""),
                ))

        decisions.append(TraceDecision(
            decision_id=f"turn_intent_{record.get('turn_index', i)}",
            decision_type="turn_intent",
            input_context=f"Turn {record.get('turn_index', i)}: "
                         f"{record.get('speaker_name', 'Unknown')} ({record.get('speaker_role', 'unknown')}): "
                         f"{record.get('text_snippet', '')[:100]}...",
            output_decision=f"Intent: {record.get('intent', 'unknown')}, "
                          f"Supports QA: {record.get('supports_qa', False)}",
            confidence=record.get("confidence"),
            reasoning=record.get("follow_up_reason", ""),
            evidence_spans=evidence_spans,
        ))

    # Q&A construction decisions
    for i, construction in enumerate(trace.get("qa_constructions", [])):
        evidence_spans = []
        for span in construction.get("evidence_spans", []):
            if isinstance(span, dict):
                evidence_spans.append(EvidenceSpanResponse(
                    text=span.get("text", "")[:300],
                    source=span.get("source", "unknown"),
                    relevance=span.get("relevance", ""),
                ))

        decisions.append(TraceDecision(
            decision_id=f"qa_construction_{construction.get('qa_id', i)}",
            decision_type="qa_construction",
            input_context=f"Q turns: {construction.get('question_turn_indices', [])}, "
                         f"A turns: {construction.get('answer_turn_indices', [])}",
            output_decision=f"Questioner: {construction.get('questioner_name', 'Unknown')}, "
                          f"Responders: {construction.get('responder_names', [])}",
            reasoning=f"Follow-up: {construction.get('is_follow_up', False)}, "
                     f"Reason: {construction.get('follow_up_reason', 'N/A')}",
            evidence_spans=evidence_spans,
        ))

    return decisions


def _extract_decisions_from_boundary_trace(trace: dict) -> list[TraceDecision]:
    """Extract decisions from boundary detection trace."""
    decisions = []

    # Boundary candidates
    for i, candidate in enumerate(trace.get("candidates_generated", [])):
        decisions.append(TraceDecision(
            decision_id=f"boundary_{i}",
            decision_type="boundary_detection",
            input_context=f"Text: {candidate.get('trigger_text', '')[:100]}...",
            output_decision=f"Section: {candidate.get('section_type', 'unknown')}, "
                          f"Confidence: {candidate.get('confidence', 0):.2f}",
            confidence=candidate.get("confidence"),
            reasoning=candidate.get("detection_method", ""),
            evidence_spans=[],
        ))

    return decisions


@router.get("/runs/{run_id}/traces", response_model=TracesResponse)
async def get_traces(
    run_id: str,
    stage: Optional[str] = Query(default=None, description="Filter by stage name"),
) -> TracesResponse:
    """
    Get LLM decision traces for a run.

    Shows all LLM decisions with their context, output, and evidence.
    This is essential for debugging and understanding pipeline behavior.

    Args:
        run_id: The run to get traces for
        stage: Optional filter for a specific stage (boundary, speakers, qa)

    Returns:
        TracesResponse with all stage traces
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    all_traces = await storage.load_traces(run_id)

    stages = []
    total_llm_calls = 0

    # Process each stage trace
    for stage_name, trace_data in all_traces.items():
        if stage and stage_name != stage:
            continue

        if not trace_data:
            continue

        llm_calls = trace_data.get("llm_calls_made", 0)
        total_llm_calls += llm_calls

        # Extract decisions based on stage type
        if stage_name == "speakers":
            decisions = _extract_decisions_from_speakers_trace(trace_data)
        elif stage_name == "qa":
            decisions = _extract_decisions_from_qa_trace(trace_data)
        elif stage_name == "boundary":
            decisions = _extract_decisions_from_boundary_trace(trace_data)
        else:
            decisions = []

        # Get hard rules
        hard_rules = trace_data.get("hard_rule_enforcements", [])

        # Get warnings
        warnings = []
        if isinstance(trace_data.get("sections_with_no_qa"), list):
            for section_id in trace_data["sections_with_no_qa"]:
                warnings.append(f"No Q&A extracted from section: {section_id}")

        stages.append(StageTrace(
            stage_name=stage_name,
            stage_type=stage_name,
            llm_calls_made=llm_calls,
            decisions=decisions,
            hard_rules_enforced=hard_rules if isinstance(hard_rules, list) else [],
            warnings=warnings,
        ))

    return TracesResponse(
        run_id=run_id,
        stages=stages,
        total_llm_calls=total_llm_calls,
    )


@router.get("/runs/{run_id}/traces/{stage_name}")
async def get_stage_trace(run_id: str, stage_name: str) -> dict:
    """
    Get raw trace data for a specific stage.

    Returns the complete trace as stored, without transformation.
    Useful for advanced debugging.
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    trace = await storage.load_run_file(run_id, f"stage_{stage_name}_trace.json")

    if not trace:
        raise HTTPException(
            status_code=404,
            detail=f"Trace not found for stage: {stage_name}"
        )

    return {
        "run_id": run_id,
        "stage_name": stage_name,
        "trace": trace,
    }
