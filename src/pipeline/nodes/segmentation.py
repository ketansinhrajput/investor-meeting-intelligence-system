"""Conversation segmentation pipeline node."""

import uuid

import structlog

from src.llm.chains import LLMChainError, run_segmentation_chain
from src.models import CallPhaseType, DocumentChunk, ErrorSeverity, SpeakerRole
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def segment_chunks_node(state: PipelineState) -> PipelineState:
    """Segment all chunks to identify speaker turns and phases.

    Processes each chunk through the LLM segmentation chain.

    Args:
        state: Current pipeline state with chunks.

    Returns:
        Updated state with segmented_chunks.
    """
    logger.info("segmentation_node_start")

    chunks_data = state.get("chunks", [])

    if not chunks_data:
        logger.warning("segmentation_node_skip_no_chunks")
        return state

    segmented_chunks = []
    errors = []
    llm_calls = state.get("llm_calls_count", 0)

    total_chunks = len(chunks_data)
    previous_speaker = None
    previous_phase = None

    for chunk_dict in chunks_data:
        chunk = DocumentChunk.model_validate(chunk_dict)

        try:
            # Run segmentation chain
            result = run_segmentation_chain(
                chunk_text=chunk.text,
                chunk_index=chunk.chunk_index,
                total_chunks=total_chunks,
                start_page=chunk.start_page,
                previous_speaker=previous_speaker,
                previous_phase=previous_phase,
            )
            llm_calls += 1

            # Build segmented chunk
            segmented = _build_segmented_chunk(
                chunk=chunk,
                llm_result=result,
            )

            segmented_chunks.append(segmented)

            # Update context for next chunk
            if segmented.get("turns"):
                last_turn = segmented["turns"][-1]
                previous_speaker = last_turn.get("speaker_name")

            if segmented.get("detected_phases"):
                previous_phase = segmented["detected_phases"][-1].get("phase_type")

            logger.debug(
                "chunk_segmented",
                chunk_index=chunk.chunk_index,
                turns=len(segmented.get("turns", [])),
            )

        except LLMChainError as e:
            logger.error(
                "segmentation_chain_error",
                chunk_index=chunk.chunk_index,
                error=str(e),
            )
            errors.append({
                "error_id": f"seg_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.ERROR.value,
                "stage": "segmentation",
                "message": str(e),
                "details": {"chunk_index": chunk.chunk_index},
                "recoverable": True,
            })

        except Exception as e:
            logger.exception(
                "segmentation_unexpected_error",
                chunk_index=chunk.chunk_index,
            )
            errors.append({
                "error_id": f"seg_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.ERROR.value,
                "stage": "segmentation",
                "message": f"Unexpected error: {e}",
                "details": {
                    "chunk_index": chunk.chunk_index,
                    "exception_type": type(e).__name__,
                },
                "recoverable": True,
            })

    logger.info(
        "segmentation_node_complete",
        chunks_processed=len(segmented_chunks),
        errors=len(errors),
    )

    return {
        **state,
        "segmented_chunks": segmented_chunks,
        "llm_calls_count": llm_calls,
        "errors": state.get("errors", []) + errors,
    }


def _build_segmented_chunk(chunk: DocumentChunk, llm_result: dict) -> dict:
    """Build a segmented chunk from LLM result.

    Args:
        chunk: Original document chunk.
        llm_result: Result from segmentation chain.

    Returns:
        SegmentedChunk as dict.
    """
    turns = []
    base_char_offset = chunk.char_offset_start

    for i, turn_data in enumerate(llm_result.get("turns", [])):
        turn_id = f"{chunk.chunk_id}_turn_{i}"

        # Normalize role
        role_str = turn_data.get("inferred_role", "unknown").lower()
        try:
            role = SpeakerRole(role_str)
        except ValueError:
            role = SpeakerRole.UNKNOWN

        turn = {
            "turn_id": turn_id,
            "speaker_name": turn_data.get("speaker_name"),
            "speaker_id": _generate_speaker_id(turn_data.get("speaker_name")),
            "inferred_role": role.value,
            "text": turn_data.get("text", ""),
            "start_char": base_char_offset,  # Approximation
            "end_char": base_char_offset + len(turn_data.get("text", "")),
            "page_number": turn_data.get("page_number", chunk.start_page),
        }
        turns.append(turn)

    # Build phases
    detected_phases = []
    for phase_data in llm_result.get("detected_phases", []):
        phase_type_str = phase_data.get("phase_type", "transition").lower()
        try:
            phase_type = CallPhaseType(phase_type_str)
        except ValueError:
            phase_type = CallPhaseType.TRANSITION

        start_idx = phase_data.get("start_turn_index", 0)
        end_idx = phase_data.get("end_turn_index", len(turns) - 1)

        if turns:
            phase = {
                "phase_type": phase_type.value,
                "start_turn_id": turns[min(start_idx, len(turns) - 1)]["turn_id"],
                "end_turn_id": turns[min(end_idx, len(turns) - 1)]["turn_id"],
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
            }
            detected_phases.append(phase)

    return {
        "chunk_id": chunk.chunk_id,
        "turns": turns,
        "detected_phases": detected_phases,
        "continuation_from_previous": chunk.overlap_with_previous > 0,
        "continues_to_next": llm_result.get("continues_to_next", False),
    }


def _generate_speaker_id(speaker_name: str | None) -> str:
    """Generate a speaker ID from name.

    Args:
        speaker_name: Speaker name or None.

    Returns:
        Normalized speaker ID.
    """
    if not speaker_name:
        return f"unknown_{uuid.uuid4().hex[:6]}"

    # Normalize: lowercase, remove special chars, replace spaces with underscore
    normalized = speaker_name.lower()
    normalized = "".join(c if c.isalnum() or c.isspace() else "" for c in normalized)
    normalized = "_".join(normalized.split())

    return normalized or f"speaker_{uuid.uuid4().hex[:6]}"
