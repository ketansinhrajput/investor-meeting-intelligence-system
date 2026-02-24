"""Text chunking pipeline node."""

import uuid

import structlog

from src.config.settings import get_settings
from src.extraction.chunker import ChunkingConfig, chunk_document
from src.models import ErrorSeverity, RawDocument
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def chunk_document_node(state: PipelineState) -> PipelineState:
    """Chunk the extracted document for LLM processing.

    Args:
        state: Current pipeline state with raw_document.

    Returns:
        Updated state with chunks list.
    """
    logger.info("chunking_node_start")

    raw_doc_dict = state.get("raw_document")

    if not raw_doc_dict:
        logger.warning("chunking_node_skip_no_document")
        return state

    try:
        # Reconstruct RawDocument from dict
        raw_doc = RawDocument.model_validate(raw_doc_dict)

        # Get chunking config from settings
        settings = get_settings()
        config = ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            min_chunk_tokens=settings.chunk_min_tokens,
            max_chunk_tokens=settings.chunk_max_tokens,
        )

        # Chunk the document
        chunks = chunk_document(raw_doc, config)

        logger.info(
            "chunking_node_complete",
            num_chunks=len(chunks),
        )

        return {
            **state,
            "chunks": [c.model_dump() for c in chunks],
            "chunks_count": len(chunks),
        }

    except Exception as e:
        logger.exception("chunking_node_error")

        error = {
            "error_id": f"chunk_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "chunking",
            "message": str(e),
            "details": {"exception_type": type(e).__name__},
            "recoverable": True,
        }

        return {
            **state,
            "chunks": [],
            "errors": state.get("errors", []) + [error],
        }
