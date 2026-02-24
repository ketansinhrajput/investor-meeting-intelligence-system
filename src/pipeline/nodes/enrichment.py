"""Enrichment pipeline node."""

import uuid

import structlog

from src.llm.chains import LLMChainError, run_enrichment_chain
from src.models import (
    ErrorSeverity,
    InvestorIntentType,
    ResponsePostureType,
    SentimentType,
)
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)


def enrich_units_node(state: PipelineState) -> PipelineState:
    """Enrich Q&A units and strategic statements with analysis.

    Adds topics, intent, posture, and evidence to each unit.

    Args:
        state: Current pipeline state with qa_units and strategic_statements.

    Returns:
        Updated state with enriched_qa_units and enriched_strategic.
    """
    logger.info("enrichment_node_start")

    qa_units = state.get("qa_units", [])
    strategic_statements = state.get("strategic_statements", [])

    enriched_qa = []
    enriched_strategic = []
    errors = []
    llm_calls = state.get("llm_calls_count", 0)

    # Enrich Q&A units
    for qa_unit in qa_units:
        try:
            enriched = _enrich_qa_unit(qa_unit)
            enriched_qa.append(enriched)
            llm_calls += 1

            logger.debug(
                "qa_unit_enriched",
                unit_id=qa_unit.get("unit_id"),
                topics=len(enriched.get("topics", [])),
            )

        except LLMChainError as e:
            logger.error(
                "qa_enrichment_error",
                unit_id=qa_unit.get("unit_id"),
                error=str(e),
            )
            errors.append({
                "error_id": f"enrich_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.WARNING.value,
                "stage": "enrichment",
                "message": f"Failed to enrich Q&A unit: {e}",
                "details": {"unit_id": qa_unit.get("unit_id")},
                "recoverable": True,
            })
            # Add unenriched version
            enriched_qa.append(_create_minimal_enriched_qa(qa_unit))

        except Exception as e:
            logger.exception("qa_enrichment_unexpected_error")
            errors.append({
                "error_id": f"enrich_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.WARNING.value,
                "stage": "enrichment",
                "message": f"Unexpected error: {e}",
                "details": {
                    "unit_id": qa_unit.get("unit_id"),
                    "exception_type": type(e).__name__,
                },
                "recoverable": True,
            })
            enriched_qa.append(_create_minimal_enriched_qa(qa_unit))

    # Enrich strategic statements
    for statement in strategic_statements:
        try:
            enriched = _enrich_strategic_statement(statement)
            enriched_strategic.append(enriched)
            llm_calls += 1

            logger.debug(
                "statement_enriched",
                statement_id=statement.get("statement_id"),
                topics=len(enriched.get("topics", [])),
            )

        except LLMChainError as e:
            logger.error(
                "statement_enrichment_error",
                statement_id=statement.get("statement_id"),
                error=str(e),
            )
            errors.append({
                "error_id": f"enrich_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.WARNING.value,
                "stage": "enrichment",
                "message": f"Failed to enrich statement: {e}",
                "details": {"statement_id": statement.get("statement_id")},
                "recoverable": True,
            })
            enriched_strategic.append(_create_minimal_enriched_statement(statement))

        except Exception as e:
            logger.exception("statement_enrichment_unexpected_error")
            errors.append({
                "error_id": f"enrich_err_{uuid.uuid4().hex[:8]}",
                "severity": ErrorSeverity.WARNING.value,
                "stage": "enrichment",
                "message": f"Unexpected error: {e}",
                "details": {
                    "statement_id": statement.get("statement_id"),
                    "exception_type": type(e).__name__,
                },
                "recoverable": True,
            })
            enriched_strategic.append(_create_minimal_enriched_statement(statement))

    logger.info(
        "enrichment_node_complete",
        enriched_qa=len(enriched_qa),
        enriched_strategic=len(enriched_strategic),
        errors=len(errors),
    )

    return {
        **state,
        "enriched_qa_units": enriched_qa,
        "enriched_strategic": enriched_strategic,
        "llm_calls_count": llm_calls,
        "errors": state.get("errors", []) + errors,
    }


def _enrich_qa_unit(qa_unit: dict) -> dict:
    """Enrich a single Q&A unit.

    Args:
        qa_unit: Q&A unit dict.

    Returns:
        EnrichedQAUnit dict.
    """
    question_text = qa_unit.get("_question_text", "")
    response_text = qa_unit.get("_response_text", "")

    # Build content for LLM
    content = f"""QUESTION:
{question_text}

RESPONSE:
{response_text}"""

    # Run enrichment chain
    result = run_enrichment_chain(
        unit_type="Q&A Unit",
        content=content,
        start_page=qa_unit.get("start_page", 1),
        end_page=qa_unit.get("end_page", 1),
    )

    # Parse intent
    intent_data = result.get("investor_intent", {})
    intent_type_str = intent_data.get("primary_intent", "exploration")
    try:
        intent_type = InvestorIntentType(intent_type_str)
    except ValueError:
        intent_type = InvestorIntentType.EXPLORATION

    # Parse posture
    posture_data = result.get("response_posture", {})
    posture_type_str = posture_data.get("primary_posture", "neutral")
    try:
        posture_type = ResponsePostureType(posture_type_str)
    except ValueError:
        posture_type = ResponsePostureType.NEUTRAL

    # Build evidence references
    key_evidence = []
    for ev in result.get("key_evidence", []):
        key_evidence.append({
            "quote": ev.get("quote", ""),
            "speaker_id": qa_unit.get("questioner_id", ""),
            "page_number": ev.get("page_number", qa_unit.get("start_page", 1)),
            "char_offset_start": None,
            "char_offset_end": None,
        })

    return {
        "unit_id": qa_unit.get("unit_id"),
        "sequence_number": qa_unit.get("sequence_number", 1),
        "question_text": question_text,
        "response_text": response_text,
        "topics": result.get("topics", []),
        "investor_intent": {
            "primary_intent": intent_type.value,
            "reasoning": intent_data.get("reasoning", ""),
        },
        "response_posture": {
            "primary_posture": posture_type.value,
            "reasoning": posture_data.get("reasoning", ""),
        },
        "key_evidence": key_evidence,
        "questioner_id": qa_unit.get("questioner_id"),
        "questioner_name": qa_unit.get("questioner_name"),
        "questioner_organization": qa_unit.get("questioner_organization"),
        "responders": qa_unit.get("responders", []),
        "start_page": qa_unit.get("start_page", 1),
        "end_page": qa_unit.get("end_page", 1),
    }


def _enrich_strategic_statement(statement: dict) -> dict:
    """Enrich a single strategic statement.

    Args:
        statement: StrategicStatement dict.

    Returns:
        EnrichedStrategicStatement dict.
    """
    text = statement.get("text", "")

    # Run enrichment chain
    result = run_enrichment_chain(
        unit_type="Strategic Statement",
        content=text,
        start_page=statement.get("start_page", 1),
        end_page=statement.get("end_page", 1),
    )

    # Determine sentiment from posture or topics
    posture_data = result.get("response_posture", {})
    posture_str = posture_data.get("primary_posture", "neutral")

    if posture_str in ("confident", "optimistic", "transparent"):
        sentiment = SentimentType.POSITIVE
    elif posture_str in ("defensive", "cautious", "evasive"):
        sentiment = SentimentType.NEGATIVE
    else:
        sentiment = SentimentType.NEUTRAL

    # Build evidence references
    key_evidence = []
    for ev in result.get("key_evidence", []):
        key_evidence.append({
            "quote": ev.get("quote", ""),
            "speaker_id": statement.get("speaker_id", ""),
            "page_number": ev.get("page_number", statement.get("start_page", 1)),
            "char_offset_start": None,
            "char_offset_end": None,
        })

    return {
        "statement_id": statement.get("statement_id"),
        "text": text,
        "summary": statement.get("_summary", "")[:200] or _generate_summary(text),
        "topics": result.get("topics", []),
        "sentiment": sentiment.value,
        "forward_looking": statement.get("_forward_looking", False),
        "key_evidence": key_evidence,
        "speaker_id": statement.get("speaker_id"),
        "speaker_name": statement.get("speaker_name"),
        "speaker_title": statement.get("speaker_title"),
        "statement_type": statement.get("statement_type"),
        "start_page": statement.get("start_page", 1),
        "end_page": statement.get("end_page", 1),
    }


def _create_minimal_enriched_qa(qa_unit: dict) -> dict:
    """Create minimal enriched Q&A when enrichment fails.

    Args:
        qa_unit: Original Q&A unit.

    Returns:
        Minimal EnrichedQAUnit dict.
    """
    return {
        "unit_id": qa_unit.get("unit_id"),
        "sequence_number": qa_unit.get("sequence_number", 1),
        "question_text": qa_unit.get("_question_text", ""),
        "response_text": qa_unit.get("_response_text", ""),
        "topics": [],
        "investor_intent": {
            "primary_intent": InvestorIntentType.EXPLORATION.value,
            "reasoning": "Unable to determine intent due to processing error.",
        },
        "response_posture": {
            "primary_posture": ResponsePostureType.NEUTRAL.value,
            "reasoning": "Unable to determine posture due to processing error.",
        },
        "key_evidence": [],
        "questioner_id": qa_unit.get("questioner_id"),
        "questioner_name": qa_unit.get("questioner_name"),
        "questioner_organization": qa_unit.get("questioner_organization"),
        "responders": qa_unit.get("responders", []),
        "start_page": qa_unit.get("start_page", 1),
        "end_page": qa_unit.get("end_page", 1),
    }


def _create_minimal_enriched_statement(statement: dict) -> dict:
    """Create minimal enriched statement when enrichment fails.

    Args:
        statement: Original statement.

    Returns:
        Minimal EnrichedStrategicStatement dict.
    """
    text = statement.get("text", "")

    return {
        "statement_id": statement.get("statement_id"),
        "text": text,
        "summary": _generate_summary(text),
        "topics": [],
        "sentiment": SentimentType.NEUTRAL.value,
        "forward_looking": statement.get("_forward_looking", False),
        "key_evidence": [],
        "speaker_id": statement.get("speaker_id"),
        "speaker_name": statement.get("speaker_name"),
        "speaker_title": statement.get("speaker_title"),
        "statement_type": statement.get("statement_type"),
        "start_page": statement.get("start_page", 1),
        "end_page": statement.get("end_page", 1),
    }


def _generate_summary(text: str, max_length: int = 150) -> str:
    """Generate a simple summary from text.

    Args:
        text: Full text.
        max_length: Maximum summary length.

    Returns:
        Summary string.
    """
    if not text:
        return ""

    # Take first sentence or truncate
    first_sentence_end = min(
        text.find(". ") + 1 if text.find(". ") > 0 else len(text),
        max_length,
    )

    summary = text[:first_sentence_end].strip()

    if len(summary) < len(text):
        summary = summary.rstrip(".") + "..."

    return summary
