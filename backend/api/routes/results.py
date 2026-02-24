"""
Results Routes

Endpoints for retrieving analysis results in various formats.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    RunListResponse,
    RunListItem,
    RunSummaryResponse,
    PipelineStageStatus,
    SpeakerRegistryResponse,
    SpeakerResponse,
    SpeakerAlias,
    QAResponse,
    QAUnitResponse,
    SpeakerTurnResponse,
    PageText,
    RawTextResponse,
    RawJsonResponse,
)
from services import storage

router = APIRouter()


# =============================================================================
# Run List
# =============================================================================

@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> RunListResponse:
    """
    List all analysis runs.

    Returns runs sorted by start time (newest first).
    """
    all_runs = await storage.list_runs()

    # Apply pagination
    paginated = all_runs[offset:offset + limit]

    items = []
    for run in paginated:
        items.append(RunListItem(
            run_id=run.get("run_id", ""),
            file_id=run.get("file_id", ""),
            filename=run.get("filename", "unknown"),
            display_name=run.get("display_name"),
            status=run.get("status", "unknown"),
            started_at=datetime.fromisoformat(run["started_at"]) if run.get("started_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(run["completed_at"]) if run.get("completed_at") else None,
            qa_count=run.get("qa_count"),
            speaker_count=run.get("speaker_count"),
            error_message=run.get("errors", [None])[0] if run.get("errors") else None,
        ))

    return RunListResponse(
        runs=items,
        total_count=len(all_runs),
    )


# =============================================================================
# Run Summary
# =============================================================================

@router.get("/runs/{run_id}/summary", response_model=RunSummaryResponse)
async def get_run_summary(run_id: str) -> RunSummaryResponse:
    """
    Get high-level summary of a run.

    Includes stats on speakers, Q&A units, pages, and pipeline stage status.
    """
    metadata = await storage.get_run_metadata(run_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Load additional data for stats
    speakers = await storage.load_speakers(run_id)
    qa = await storage.load_qa_units(run_id)
    raw_text = await storage.load_raw_text(run_id)

    # Build stage statuses
    stages = []
    stage_data = metadata.get("stages", {})
    for stage_name in ["extraction", "metadata", "boundary", "speakers", "qa", "strategic"]:
        stage_info = stage_data.get(stage_name, {})
        stages.append(PipelineStageStatus(
            stage_name=stage_name,
            status=stage_info.get("status", "pending"),
            completed_at=datetime.fromisoformat(stage_info["completed_at"]) if stage_info.get("completed_at") else None,
        ))

    # Calculate duration
    duration = None
    if metadata.get("started_at") and metadata.get("completed_at"):
        start = datetime.fromisoformat(metadata["started_at"])
        end = datetime.fromisoformat(metadata["completed_at"])
        duration = (end - start).total_seconds()

    # Count follow-ups
    follow_up_count = 0
    if qa and qa.get("qa_units"):
        follow_up_count = sum(1 for u in qa["qa_units"] if u.get("is_follow_up"))

    # Get error message
    errors = metadata.get("errors", [])
    error_message = errors[0] if errors else metadata.get("error_message")

    return RunSummaryResponse(
        run_id=run_id,
        file_id=metadata.get("file_id", ""),
        filename=metadata.get("filename", "unknown"),
        display_name=metadata.get("display_name"),
        status=metadata.get("status", "unknown"),
        started_at=datetime.fromisoformat(metadata["started_at"]) if metadata.get("started_at") else datetime.utcnow(),
        completed_at=datetime.fromisoformat(metadata["completed_at"]) if metadata.get("completed_at") else None,
        duration_seconds=duration,
        page_count=metadata.get("page_count", 0),
        total_text_length=raw_text.get("total_chars", 0) if raw_text else 0,
        speaker_count=metadata.get("speaker_count", 0),
        qa_count=metadata.get("qa_count", 0),
        follow_up_count=follow_up_count,
        strategic_statement_count=0,  # TODO: Add when available
        stages=stages,
        errors=errors,
        warnings=metadata.get("warnings", []),
        error_message=error_message,
    )


# =============================================================================
# Speakers
# =============================================================================

@router.get("/runs/{run_id}/speakers", response_model=SpeakerRegistryResponse)
async def get_speakers(run_id: str) -> SpeakerRegistryResponse:
    """
    Get the speaker registry for a run.

    Includes canonical names, roles, titles, and alias information.
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    speakers_data = await storage.load_speakers(run_id)
    trace_data = await storage.load_run_file(run_id, "stage_speakers_trace.json")

    if not speakers_data:
        return SpeakerRegistryResponse(
            run_id=run_id,
            speakers=[],
            total_count=0,
        )

    speakers_list = []
    speakers_dict = speakers_data.get("speakers", {})

    # Build verification lookup from trace
    verification_lookup = {}
    if trace_data and trace_data.get("verification_decisions"):
        for decision in trace_data["verification_decisions"]:
            name = decision.get("canonical_name", decision.get("candidate_name", ""))
            verification_lookup[name.lower()] = decision

    for speaker_id, speaker in speakers_dict.items():
        # Build aliases - filter out empty strings and invalid entries
        aliases = []
        raw_aliases = speaker.get("aliases", [])
        if isinstance(raw_aliases, list):
            for alias in raw_aliases:
                if isinstance(alias, str) and alias.strip():
                    # Only add non-empty string aliases
                    aliases.append(SpeakerAlias(alias=alias.strip()))
                elif isinstance(alias, dict):
                    alias_name = alias.get("alias", "").strip()
                    if alias_name:  # Only add if alias name is not empty
                        aliases.append(SpeakerAlias(
                            alias=alias_name,
                            merge_reason=alias.get("merge_reason"),
                            confidence=alias.get("confidence"),
                        ))

        # Get LLM verification info
        verification = verification_lookup.get(speaker.get("canonical_name", "").lower(), {})

        speakers_list.append(SpeakerResponse(
            speaker_id=speaker_id,
            canonical_name=speaker.get("canonical_name", "Unknown"),
            role=speaker.get("role", "unknown"),
            title=speaker.get("title"),
            company=speaker.get("company"),
            turn_count=speaker.get("turn_count", 0),
            first_appearance_page=speaker.get("first_appearance_page"),
            aliases=aliases,
            verified_by_llm=verification.get("verified_by_llm", False),
            llm_confidence=verification.get("role_confidence"),
            llm_reasoning=verification.get("reasoning"),
        ))

    # Sort by role priority (moderator, management, analyst, unknown) then by name
    role_order = {"moderator": 0, "management": 1, "analyst": 2, "unknown": 3}
    speakers_list.sort(key=lambda s: (role_order.get(s.role, 4), s.canonical_name))

    return SpeakerRegistryResponse(
        run_id=run_id,
        speakers=speakers_list,
        total_count=len(speakers_list),
        management_count=speakers_data.get("management_count", 0),
        analyst_count=speakers_data.get("analyst_count", 0),
        moderator_count=sum(1 for s in speakers_list if s.role == "moderator"),
    )


# =============================================================================
# Q&A Units
# =============================================================================

@router.get("/runs/{run_id}/qa", response_model=QAResponse)
async def get_qa_units(run_id: str) -> QAResponse:
    """
    Get all Q&A units for a run.

    Includes question/answer text, speaker information, and follow-up chains.
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    qa_data = await storage.load_qa_units(run_id)
    trace_data = await storage.load_run_file(run_id, "stage_qa_trace.json")

    if not qa_data:
        return QAResponse(
            run_id=run_id,
            qa_units=[],
            total_count=0,
        )

    # Build follow-up index
    follow_up_map: dict[str, list[str]] = {}
    for unit in qa_data.get("qa_units", []):
        if unit.get("follow_up_of"):
            parent_id = unit["follow_up_of"]
            if parent_id not in follow_up_map:
                follow_up_map[parent_id] = []
            follow_up_map[parent_id].append(unit.get("qa_id", ""))

    # Build trace lookup
    construction_lookup = {}
    if trace_data and trace_data.get("qa_constructions"):
        for construction in trace_data["qa_constructions"]:
            qa_id = construction.get("qa_id", "")
            construction_lookup[qa_id] = construction

    qa_units = []
    for i, unit in enumerate(qa_data.get("qa_units", [])):
        qa_id = unit.get("qa_id", f"qa_{i:03d}")

        # Build question turns
        question_turns = []
        for turn in unit.get("question_turns", []):
            question_turns.append(SpeakerTurnResponse(
                speaker_name=turn.get("speaker_name", "Unknown"),
                speaker_id=turn.get("speaker_id"),
                text=turn.get("text", ""),
                page_number=turn.get("page_number"),
                is_question=True,
            ))

        # Build response turns
        response_turns = []
        for turn in unit.get("response_turns", []):
            response_turns.append(SpeakerTurnResponse(
                speaker_name=turn.get("speaker_name", "Unknown"),
                speaker_id=turn.get("speaker_id"),
                text=turn.get("text", ""),
                page_number=turn.get("page_number"),
                is_question=False,
            ))

        # Get construction trace
        construction = construction_lookup.get(qa_id, {})

        # Get boundary reasoning from the Q&A unit itself or from trace
        boundary_reasoning = (
            unit.get("boundary_justification") or
            unit.get("boundary_reasoning") or
            (construction.get("follow_up_reason") if construction else None)
        )

        qa_units.append(QAUnitResponse(
            qa_id=qa_id,
            sequence=unit.get("sequence_in_session", i),
            questioner_name=unit.get("questioner_name", "Unknown"),
            questioner_id=unit.get("questioner_id"),
            questioner_company=unit.get("questioner_company"),
            question_text=unit.get("question_text", ""),
            question_turns=question_turns,
            responder_names=unit.get("responder_names", []),
            responder_ids=unit.get("responder_ids", []),
            response_text=unit.get("response_text", ""),
            response_turns=response_turns,
            is_follow_up=unit.get("is_follow_up", False),
            follow_up_of=unit.get("follow_up_of"),
            has_follow_ups=qa_id in follow_up_map,
            follow_up_ids=follow_up_map.get(qa_id, []),
            start_page=unit.get("start_page"),
            end_page=unit.get("end_page"),
            source_section_id=unit.get("source_section_id"),
            # Enrichment data
            topics=unit.get("topics", []),
            investor_intent=unit.get("investor_intent"),
            response_posture=unit.get("response_posture"),
            boundary_reasoning=boundary_reasoning,
        ))

    return QAResponse(
        run_id=run_id,
        qa_units=qa_units,
        total_count=len(qa_units),
        follow_up_count=qa_data.get("total_follow_ups", 0),
        unique_questioners=qa_data.get("unique_questioners", 0),
    )


# =============================================================================
# Raw Text
# =============================================================================

@router.get("/runs/{run_id}/raw", response_model=RawTextResponse)
async def get_raw_text(run_id: str) -> RawTextResponse:
    """
    Get raw extracted text per page.

    Useful for debugging extraction issues.
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    raw_data = await storage.load_raw_text(run_id)

    if not raw_data:
        return RawTextResponse(
            run_id=run_id,
            pages=[],
            total_pages=0,
        )

    pages = [
        PageText(
            page_number=p.get("page_number", i + 1),
            text=p.get("text", ""),
            char_count=p.get("char_count", len(p.get("text", ""))),
        )
        for i, p in enumerate(raw_data.get("pages", []))
    ]

    return RawTextResponse(
        run_id=run_id,
        pages=pages,
        total_pages=raw_data.get("total_pages", len(pages)),
        total_chars=raw_data.get("total_chars", 0),
    )


# =============================================================================
# Raw JSON
# =============================================================================

@router.get("/runs/{run_id}/json", response_model=RawJsonResponse)
async def get_raw_json(run_id: str) -> RawJsonResponse:
    """
    Get the complete pipeline output as raw JSON.

    Useful for debugging and development.
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    pipeline_output = await storage.load_pipeline_output(run_id)

    # Also load all stage outputs
    stages_output = {}
    for stage in ["extraction", "metadata", "boundary", "speakers", "qa", "strategic"]:
        result = await storage.load_run_file(run_id, f"stage_{stage}_result.json")
        if result:
            stages_output[stage] = result

    return RawJsonResponse(
        run_id=run_id,
        pipeline_output=pipeline_output or {},
        stages_output=stages_output,
    )


# =============================================================================
# Delete Run
# =============================================================================

@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    """
    Delete a run and all its associated files.

    This permanently removes:
    - All stage results
    - All traces
    - Metadata
    - Pipeline output
    """
    if not await storage.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    deleted = await storage.delete_run(run_id)

    if deleted:
        return {"status": "deleted", "run_id": run_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete run")
