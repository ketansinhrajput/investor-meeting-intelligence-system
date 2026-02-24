"""PDF extraction pipeline node."""

import uuid

import structlog

from src.extraction.pdf_extractor import PDFExtractionError, extract_pdf
from src.models import ErrorSeverity
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def extract_pdf_node(state: PipelineState) -> PipelineState:
    """Extract text from PDF file.

    Args:
        state: Current pipeline state with pdf_path.

    Returns:
        Updated state with raw_document or error.
    """
    pdf_path = state["pdf_path"]
    logger.info("pdf_extraction_node_start", path=pdf_path)

    try:
        raw_doc = extract_pdf(pdf_path)

        logger.info(
            "pdf_extraction_node_complete",
            pages=raw_doc.total_pages,
            chars=raw_doc.total_characters,
        )

        return {
            **state,
            "raw_document": raw_doc.model_dump(),
        }

    except PDFExtractionError as e:
        logger.error("pdf_extraction_node_failed", error=str(e))

        error = {
            "error_id": f"pdf_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.CRITICAL.value,
            "stage": "pdf_extraction",
            "message": str(e),
            "details": {"pdf_path": pdf_path},
            "recoverable": False,
        }

        return {
            **state,
            "raw_document": None,
            "errors": state.get("errors", []) + [error],
        }

    except Exception as e:
        logger.exception("pdf_extraction_node_unexpected_error")

        error = {
            "error_id": f"pdf_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.CRITICAL.value,
            "stage": "pdf_extraction",
            "message": f"Unexpected error: {e}",
            "details": {"pdf_path": pdf_path, "exception_type": type(e).__name__},
            "recoverable": False,
        }

        return {
            **state,
            "raw_document": None,
            "errors": state.get("errors", []) + [error],
        }
