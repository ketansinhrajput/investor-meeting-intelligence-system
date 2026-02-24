"""Q&A unit extraction pipeline node."""

import json
import uuid

import structlog

from src.llm.chains import LLMChainError, run_qa_extraction_chain
from src.models import ErrorSeverity, SpeakerRole
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def extract_qa_units_node(state: PipelineState) -> PipelineState:
    """Extract Q&A units from segmented transcript.

    Identifies question-response pairs from investor/analyst
    and management speaker turns.

    Args:
        state: Current pipeline state with segmented_transcript.

    Returns:
        Updated state with qa_units.
    """
    logger.info("qa_extraction_node_start")

    transcript = state.get("segmented_transcript")

    if not transcript:
        logger.warning("qa_extraction_node_skip_no_transcript")
        return state

    try:
        turns = transcript.get("turns", [])
        phases = transcript.get("phases", [])

        # Filter turns in Q&A phase
        qa_turns = _get_qa_phase_turns(turns, phases)

        if not qa_turns:
            logger.info("qa_extraction_node_no_qa_turns")
            return {
                **state,
                "qa_units": [],
            }

        # Prepare turns JSON for LLM
        turns_for_llm = _prepare_turns_for_llm(qa_turns)
        turns_json = json.dumps(turns_for_llm, indent=2)

        # Run extraction chain
        llm_result = run_qa_extraction_chain(turns_json)
        llm_calls = state.get("llm_calls_count", 0) + 1

        # Build Q&A units from result
        qa_units = _build_qa_units(
            llm_result=llm_result,
            turns=qa_turns,
            speaker_registry=state.get("speaker_registry", {}),
        )

        logger.info(
            "qa_extraction_node_complete",
            qa_units=len(qa_units),
        )

        return {
            **state,
            "qa_units": qa_units,
            "llm_calls_count": llm_calls,
        }

    except LLMChainError as e:
        logger.error("qa_extraction_chain_error", error=str(e))

        error = {
            "error_id": f"qa_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "qa_extraction",
            "message": str(e),
            "details": None,
            "recoverable": True,
        }

        return {
            **state,
            "qa_units": [],
            "errors": state.get("errors", []) + [error],
        }

    except Exception as e:
        logger.exception("qa_extraction_unexpected_error")

        error = {
            "error_id": f"qa_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "qa_extraction",
            "message": f"Unexpected error: {e}",
            "details": {"exception_type": type(e).__name__},
            "recoverable": True,
        }

        return {
            **state,
            "qa_units": [],
            "errors": state.get("errors", []) + [error],
        }


def _get_qa_phase_turns(turns: list[dict], phases: list[dict]) -> list[dict]:
    """Get turns that are in Q&A phase.

    Args:
        turns: All turns.
        phases: Detected phases.

    Returns:
        Turns within Q&A phase.
    """
    # Find Q&A phase boundaries
    qa_phase = None
    for phase in phases:
        if phase.get("phase_type") == "qa_session":
            qa_phase = phase
            break

    if not qa_phase:
        # No explicit Q&A phase, try to infer from speaker roles
        # Look for first investor/analyst turn
        qa_start_idx = None
        for i, turn in enumerate(turns):
            if turn.get("inferred_role") == SpeakerRole.INVESTOR_ANALYST.value:
                qa_start_idx = i
                break

        if qa_start_idx is not None:
            return turns[qa_start_idx:]
        return []

    # Get turn IDs in Q&A phase
    start_id = qa_phase.get("start_turn_id")
    end_id = qa_phase.get("end_turn_id")

    # Find indices
    start_idx = 0
    end_idx = len(turns)

    for i, turn in enumerate(turns):
        if turn.get("turn_id") == start_id:
            start_idx = i
        if turn.get("turn_id") == end_id:
            end_idx = i + 1

    return turns[start_idx:end_idx]


def _prepare_turns_for_llm(turns: list[dict]) -> list[dict]:
    """Prepare turns for LLM processing.

    Simplifies turn data for cleaner LLM input.

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
            "text_preview": turn.get("text", "")[:500],  # Truncate for context
            "page": turn.get("page_number"),
        }
        for i, turn in enumerate(turns)
    ]


def _build_qa_units(
    llm_result: dict,
    turns: list[dict],
    speaker_registry: dict,
) -> list[dict]:
    """Build Q&A unit objects from LLM result.

    Args:
        llm_result: Result from Q&A extraction chain.
        turns: Original turns.
        speaker_registry: Speaker registry.

    Returns:
        List of QAUnit dicts.
    """
    qa_units = []
    speakers = speaker_registry.get("speakers", {})

    for seq, unit_data in enumerate(llm_result.get("qa_units", []), start=1):
        q_indices = unit_data.get("question_turn_indices", [])
        r_indices = unit_data.get("response_turn_indices", [])

        if not q_indices:
            continue

        # Get question turns
        question_turns = [turns[i] for i in q_indices if 0 <= i < len(turns)]
        response_turns = [turns[i] for i in r_indices if 0 <= i < len(turns)]

        if not question_turns:
            continue

        # Get questioner info
        questioner_id = unit_data.get("questioner_speaker_id") or question_turns[0].get("speaker_id")
        questioner_profile = speakers.get(questioner_id, {})

        # Get responder IDs
        responder_ids = unit_data.get("responder_speaker_ids", [])
        if not responder_ids and response_turns:
            responder_ids = list(set(t.get("speaker_id") for t in response_turns))

        # Calculate page range
        all_unit_turns = question_turns + response_turns
        pages = [t.get("page_number", 1) for t in all_unit_turns]
        start_page = min(pages) if pages else 1
        end_page = max(pages) if pages else 1

        qa_unit = {
            "unit_id": f"qa_{uuid.uuid4().hex[:8]}",
            "sequence_number": seq,
            "question_turns": [t.get("turn_id") for t in question_turns],
            "questioner_id": questioner_id,
            "questioner_name": questioner_profile.get("canonical_name") or question_turns[0].get("speaker_name"),
            "questioner_organization": questioner_profile.get("organization"),
            "response_turns": [t.get("turn_id") for t in response_turns],
            "responders": responder_ids,
            "start_page": start_page,
            "end_page": end_page,
            "moderator_introduction": None,
            "moderator_turn_id": None,
            # Store full text for enrichment
            "_question_text": " ".join(t.get("text", "") for t in question_turns),
            "_response_text": " ".join(t.get("text", "") for t in response_turns),
        }

        qa_units.append(qa_unit)

    return qa_units
