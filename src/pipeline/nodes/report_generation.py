"""Report generation pipeline node."""

import uuid
from datetime import datetime

import structlog

from src.llm.chains import run_metadata_extraction_chain
from src.models import ErrorSeverity, RawDocument
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

PIPELINE_VERSION = "0.1.0"


def generate_report_node(state: PipelineState) -> PipelineState:
    """Generate the final structured report.

    Aggregates all extracted and enriched data into the
    final StructuredReport format.

    Args:
        state: Current pipeline state with all extracted data.

    Returns:
        Updated state with final report.
    """
    logger.info("report_generation_node_start")

    try:
        # Extract metadata from transcript if available
        call_metadata = _extract_call_metadata(state)

        # Aggregate topics
        topic_summaries = _aggregate_topics(state)

        # Build processing metadata
        processing_metadata = _build_processing_metadata(state)

        # Get call phases
        call_phases = []
        transcript = state.get("segmented_transcript")
        if transcript:
            call_phases = transcript.get("phases", [])

        # Build final report
        report = {
            "report_id": str(uuid.uuid4()),
            "report_version": "1.0.0",
            "call_metadata": call_metadata,
            "processing_metadata": processing_metadata,
            "speaker_registry": state.get("speaker_registry", {"speakers": {}}),
            "qa_units": state.get("enriched_qa_units", []),
            "strategic_statements": state.get("enriched_strategic", []),
            "topic_summaries": topic_summaries,
            "call_phases": call_phases,
        }

        logger.info(
            "report_generation_node_complete",
            qa_units=len(report["qa_units"]),
            strategic_statements=len(report["strategic_statements"]),
            topics=len(report["topic_summaries"]),
        )

        return {
            **state,
            "report": report,
            "call_metadata": call_metadata,
            "topic_summaries": topic_summaries,
        }

    except Exception as e:
        logger.exception("report_generation_error")

        error = {
            "error_id": f"report_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "report_generation",
            "message": str(e),
            "details": {"exception_type": type(e).__name__},
            "recoverable": False,
        }

        # Return partial report with errors
        return {
            **state,
            "report": _build_error_report(state, error),
            "errors": state.get("errors", []) + [error],
        }


def _extract_call_metadata(state: PipelineState) -> dict:
    """Extract call metadata from transcript.

    Args:
        state: Pipeline state.

    Returns:
        CallMetadata dict.
    """
    raw_doc_dict = state.get("raw_document")

    if not raw_doc_dict:
        return {
            "company_name": None,
            "ticker_symbol": None,
            "fiscal_quarter": None,
            "fiscal_year": None,
            "call_date": None,
            "source_file": state.get("pdf_path", "unknown"),
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "total_pages": 0,
            "participant_count": 0,
            "qa_unit_count": 0,
            "strategic_statement_count": 0,
        }

    raw_doc = RawDocument.model_validate(raw_doc_dict)

    # Try to extract metadata from first pages
    metadata = {
        "company_name": None,
        "ticker_symbol": None,
        "fiscal_quarter": None,
        "fiscal_year": None,
        "call_date": None,
    }

    # Get text from first 2 pages for metadata extraction
    first_pages_text = ""
    for page in raw_doc.pages[:2]:
        first_pages_text += page.text + "\n"

    if first_pages_text.strip():
        try:
            extracted = run_metadata_extraction_chain(first_pages_text[:3000])
            metadata.update({
                "company_name": extracted.get("company_name"),
                "ticker_symbol": extracted.get("ticker_symbol"),
                "fiscal_quarter": extracted.get("fiscal_quarter"),
                "fiscal_year": extracted.get("fiscal_year"),
                "call_date": extracted.get("call_date"),
            })
        except Exception as e:
            logger.warning("metadata_extraction_failed", error=str(e))

    # Count participants from speaker registry
    speaker_registry = state.get("speaker_registry", {})
    participant_count = len(speaker_registry.get("speakers", {}))

    return {
        **metadata,
        "source_file": raw_doc.source_file,
        "extraction_timestamp": raw_doc.extraction_timestamp.isoformat(),
        "total_pages": raw_doc.total_pages,
        "participant_count": participant_count,
        "qa_unit_count": len(state.get("enriched_qa_units", [])),
        "strategic_statement_count": len(state.get("enriched_strategic", [])),
    }


def _build_processing_metadata(state: PipelineState) -> dict:
    """Build processing metadata.

    Args:
        state: Pipeline state.

    Returns:
        ProcessingMetadata dict.
    """
    from src.config.settings import get_settings

    settings = get_settings()

    processing_start = state.get("processing_start", datetime.utcnow().isoformat())
    processing_end = datetime.utcnow()

    # Calculate duration
    try:
        start_dt = datetime.fromisoformat(processing_start)
        duration = (processing_end - start_dt).total_seconds()
    except Exception:
        duration = 0.0

    return {
        "pipeline_version": PIPELINE_VERSION,
        "processing_start": processing_start,
        "processing_end": processing_end.isoformat(),
        "total_duration_seconds": duration,
        "llm_model": settings.llm_model_name,
        "llm_temperature": settings.llm_temperature,
        "chunks_processed": state.get("chunks_count", 0),
        "llm_calls_made": state.get("llm_calls_count", 0),
        "errors": state.get("errors", []),
        "warnings": state.get("warnings", []),
    }


def _aggregate_topics(state: PipelineState) -> list[dict]:
    """Aggregate topics across all units.

    Args:
        state: Pipeline state.

    Returns:
        List of TopicSummary dicts.
    """
    topic_data: dict[str, dict] = {}

    # Collect topics from Q&A units
    for qa in state.get("enriched_qa_units", []):
        unit_id = qa.get("unit_id")
        for topic in qa.get("topics", []):
            topic_name = topic.get("topic_name", "").lower()
            if not topic_name:
                continue

            if topic_name not in topic_data:
                topic_data[topic_name] = {
                    "topic_name": topic.get("topic_name"),
                    "topic_category": topic.get("topic_category", "General"),
                    "qa_unit_ids": [],
                    "statement_ids": [],
                    "evidence_spans": [],
                }

            topic_data[topic_name]["qa_unit_ids"].append(unit_id)
            topic_data[topic_name]["evidence_spans"].extend(
                topic.get("evidence_spans", [])
            )

    # Collect topics from strategic statements
    for stmt in state.get("enriched_strategic", []):
        statement_id = stmt.get("statement_id")
        for topic in stmt.get("topics", []):
            topic_name = topic.get("topic_name", "").lower()
            if not topic_name:
                continue

            if topic_name not in topic_data:
                topic_data[topic_name] = {
                    "topic_name": topic.get("topic_name"),
                    "topic_category": topic.get("topic_category", "General"),
                    "qa_unit_ids": [],
                    "statement_ids": [],
                    "evidence_spans": [],
                }

            topic_data[topic_name]["statement_ids"].append(statement_id)
            topic_data[topic_name]["evidence_spans"].extend(
                topic.get("evidence_spans", [])
            )

    # Build topic summaries
    summaries = []
    for topic_name, data in topic_data.items():
        mention_count = len(data["qa_unit_ids"]) + len(data["statement_ids"])

        summary = {
            "topic_name": data["topic_name"],
            "topic_category": data["topic_category"],
            "mention_count": mention_count,
            "qa_unit_ids": list(set(data["qa_unit_ids"])),
            "statement_ids": list(set(data["statement_ids"])),
            "summary": f"Topic discussed in {mention_count} unit(s).",
            "key_points": data["evidence_spans"][:5],  # Top 5 evidence spans
            "sentiment_distribution": {},
        }
        summaries.append(summary)

    # Sort by mention count
    summaries.sort(key=lambda x: x["mention_count"], reverse=True)

    return summaries


def _build_error_report(state: PipelineState, error: dict) -> dict:
    """Build a minimal error report when generation fails.

    Args:
        state: Pipeline state.
        error: The error that occurred.

    Returns:
        Minimal report dict.
    """
    return {
        "report_id": str(uuid.uuid4()),
        "report_version": "1.0.0",
        "call_metadata": {
            "company_name": None,
            "ticker_symbol": None,
            "fiscal_quarter": None,
            "fiscal_year": None,
            "call_date": None,
            "source_file": state.get("pdf_path", "unknown"),
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "total_pages": 0,
            "participant_count": 0,
            "qa_unit_count": 0,
            "strategic_statement_count": 0,
        },
        "processing_metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "processing_start": state.get("processing_start", datetime.utcnow().isoformat()),
            "processing_end": datetime.utcnow().isoformat(),
            "total_duration_seconds": 0.0,
            "llm_model": "unknown",
            "llm_temperature": 0.0,
            "chunks_processed": 0,
            "llm_calls_made": 0,
            "errors": state.get("errors", []) + [error],
            "warnings": [],
        },
        "speaker_registry": {"speakers": {}},
        "qa_units": [],
        "strategic_statements": [],
        "topic_summaries": [],
        "call_phases": [],
    }
