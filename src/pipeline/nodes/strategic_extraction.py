"""Strategic statement extraction pipeline node."""

import json
import uuid

import structlog

from src.llm.chains import LLMChainError, run_strategic_extraction_chain
from src.models import CallPhaseType, ErrorSeverity, SpeakerRole, StatementType
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def extract_strategic_statements_node(state: PipelineState) -> PipelineState:
    """Extract strategic statements from transcript.

    Identifies significant management statements in opening/closing
    remarks and notable Q&A responses.

    Args:
        state: Current pipeline state with segmented_transcript.

    Returns:
        Updated state with strategic_statements.
    """
    logger.info("strategic_extraction_node_start")

    transcript = state.get("segmented_transcript")

    if not transcript:
        logger.warning("strategic_extraction_node_skip_no_transcript")
        return state

    try:
        turns = transcript.get("turns", [])
        phases = transcript.get("phases", [])

        # Get turns from opening and closing remarks
        strategic_turns = _get_strategic_phase_turns(turns, phases)

        if not strategic_turns:
            logger.info("strategic_extraction_node_no_strategic_turns")
            return {
                **state,
                "strategic_statements": [],
            }

        # Prepare turns JSON for LLM
        turns_for_llm = _prepare_turns_for_llm(strategic_turns)
        turns_json = json.dumps(turns_for_llm, indent=2)

        # Run extraction chain
        llm_result = run_strategic_extraction_chain(turns_json)
        llm_calls = state.get("llm_calls_count", 0) + 1

        # Build strategic statements from result
        statements = _build_strategic_statements(
            llm_result=llm_result,
            turns=strategic_turns,
            speaker_registry=state.get("speaker_registry", {}),
        )

        logger.info(
            "strategic_extraction_node_complete",
            statements=len(statements),
        )

        return {
            **state,
            "strategic_statements": statements,
            "llm_calls_count": llm_calls,
        }

    except LLMChainError as e:
        logger.error("strategic_extraction_chain_error", error=str(e))

        error = {
            "error_id": f"strat_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "strategic_extraction",
            "message": str(e),
            "details": None,
            "recoverable": True,
        }

        return {
            **state,
            "strategic_statements": [],
            "errors": state.get("errors", []) + [error],
        }

    except Exception as e:
        logger.exception("strategic_extraction_unexpected_error")

        error = {
            "error_id": f"strat_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "strategic_extraction",
            "message": f"Unexpected error: {e}",
            "details": {"exception_type": type(e).__name__},
            "recoverable": True,
        }

        return {
            **state,
            "strategic_statements": [],
            "errors": state.get("errors", []) + [error],
        }


def _get_strategic_phase_turns(turns: list[dict], phases: list[dict]) -> list[dict]:
    """Get turns from opening/closing remarks phases.

    Also includes management turns that may contain strategic content.

    Args:
        turns: All turns.
        phases: Detected phases.

    Returns:
        Turns to analyze for strategic statements.
    """
    strategic_turns = []
    turn_ids_added = set()

    # Get turns from opening remarks
    for phase in phases:
        phase_type = phase.get("phase_type")
        if phase_type in ("opening_remarks", "closing_remarks"):
            start_id = phase.get("start_turn_id")
            end_id = phase.get("end_turn_id")

            in_phase = False
            for turn in turns:
                turn_id = turn.get("turn_id")

                if turn_id == start_id:
                    in_phase = True

                if in_phase and turn_id not in turn_ids_added:
                    # Only include management turns
                    if turn.get("inferred_role") == SpeakerRole.MANAGEMENT.value:
                        strategic_turns.append(turn)
                        turn_ids_added.add(turn_id)

                if turn_id == end_id:
                    in_phase = False
                    break

    # If no explicit phases, get management turns from start of transcript
    if not strategic_turns:
        for turn in turns:
            if turn.get("inferred_role") == SpeakerRole.MANAGEMENT.value:
                strategic_turns.append(turn)
                turn_ids_added.add(turn.get("turn_id"))
            elif turn.get("inferred_role") == SpeakerRole.INVESTOR_ANALYST.value:
                # Stop when Q&A starts
                break

    return strategic_turns


def _prepare_turns_for_llm(turns: list[dict]) -> list[dict]:
    """Prepare turns for LLM processing.

    Args:
        turns: Turn dicts.

    Returns:
        Simplified turn list for LLM.
    """
    return [
        {
            "index": i,
            "turn_id": turn.get("turn_id"),
            "speaker_id": turn.get("speaker_id"),
            "speaker_name": turn.get("speaker_name"),
            "role": turn.get("inferred_role"),
            "text": turn.get("text", ""),
            "page": turn.get("page_number"),
        }
        for i, turn in enumerate(turns)
    ]


def _build_strategic_statements(
    llm_result: dict,
    turns: list[dict],
    speaker_registry: dict,
) -> list[dict]:
    """Build strategic statement objects from LLM result.

    Args:
        llm_result: Result from strategic extraction chain.
        turns: Original turns.
        speaker_registry: Speaker registry.

    Returns:
        List of StrategicStatement dicts.
    """
    statements = []
    speakers = speaker_registry.get("speakers", {})

    for stmt_data in llm_result.get("strategic_statements", []):
        turn_indices = stmt_data.get("turn_indices", [])

        if not turn_indices:
            continue

        # Get statement turns
        stmt_turns = [turns[i] for i in turn_indices if 0 <= i < len(turns)]

        if not stmt_turns:
            continue

        # Get speaker info from first turn
        first_turn = stmt_turns[0]
        speaker_id = first_turn.get("speaker_id")
        speaker_profile = speakers.get(speaker_id, {})

        # Determine statement type
        stmt_type_str = stmt_data.get("statement_type", "other")
        try:
            stmt_type = StatementType(stmt_type_str)
        except ValueError:
            stmt_type = StatementType.OTHER

        # Determine phase
        # For now, assume opening_remarks if near start of document
        pages = [t.get("page_number", 1) for t in stmt_turns]
        avg_page = sum(pages) / len(pages) if pages else 1

        if avg_page <= 3:
            phase = CallPhaseType.OPENING_REMARKS
        else:
            phase = CallPhaseType.CLOSING_REMARKS

        # Build statement text
        statement_text = " ".join(t.get("text", "") for t in stmt_turns)

        statement = {
            "statement_id": f"strat_{uuid.uuid4().hex[:8]}",
            "speaker_id": speaker_id,
            "speaker_name": speaker_profile.get("canonical_name") or first_turn.get("speaker_name", "Unknown"),
            "speaker_title": speaker_profile.get("title"),
            "phase": phase.value,
            "text": statement_text,
            "turn_ids": [t.get("turn_id") for t in stmt_turns],
            "statement_type": stmt_type.value,
            "start_page": min(pages) if pages else 1,
            "end_page": max(pages) if pages else 1,
            # Additional data for enrichment
            "_summary": stmt_data.get("summary", ""),
            "_forward_looking": stmt_data.get("forward_looking", False),
        }

        statements.append(statement)

    return statements
