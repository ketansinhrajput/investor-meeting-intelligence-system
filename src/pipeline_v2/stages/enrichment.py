"""Stage 6: Enrichment - Add topics, intent, posture, and evidence to extracted content.

Responsibilities:
- Identify topics mentioned in Q&A and strategic statements
- Analyze investor intent (concern, clarification, validation, etc.)
- Analyze management response posture (confident, cautious, defensive, etc.)
- Extract key evidence spans with page references
"""

from typing import Optional

import structlog

from src.llm.chains import run_enrichment_chain
from src.pipeline_v2.models import (
    EnrichedQAUnit,
    EnrichedStrategicStatement,
    EvidenceSpan,
    ExtractedQAUnit,
    ExtractedStrategicStatement,
    InvestorIntent,
    QAExtractionResult,
    ResponsePosture,
    StrategicExtractionResult,
    TopicMention,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Enrichment Functions
# =============================================================================

def _copy_base_fields(qa: ExtractedQAUnit) -> dict:
    """Copy all base fields from ExtractedQAUnit for EnrichedQAUnit."""
    return {
        "qa_id": qa.qa_id,
        "source_section_id": qa.source_section_id,
        "questioner_name": qa.questioner_name,
        "questioner_id": qa.questioner_id,
        "question_turns": qa.question_turns,
        "response_turns": qa.response_turns,
        "responder_names": qa.responder_names,
        "responder_ids": qa.responder_ids,
        "question_text": qa.question_text,
        "response_text": qa.response_text,
        "is_follow_up": qa.is_follow_up,
        "follow_up_of": qa.follow_up_of,
        "start_page": qa.start_page,
        "end_page": qa.end_page,
        "sequence_in_session": qa.sequence_in_session,
        "boundary_justification": qa.boundary_justification,
    }


def _enrich_single_qa(qa: ExtractedQAUnit) -> EnrichedQAUnit:
    """Enrich a single Q&A unit with LLM analysis."""
    base_fields = _copy_base_fields(qa)

    # Prepare content for enrichment
    content = f"""Question ({qa.questioner_name}):
{qa.question_text}

Response ({', '.join(qa.responder_names) or 'Management'}):
{qa.response_text}"""

    try:
        enrichment = run_enrichment_chain(
            unit_type="Q&A Unit",
            content=content,
            start_page=qa.start_page,
            end_page=qa.end_page,
        )

        # Parse topics
        topics = []
        for topic_data in enrichment.get("topics", []):
            topics.append(TopicMention(
                topic_name=topic_data.get("name", "Unknown"),
                topic_category=topic_data.get("category", "Other"),
                evidence_spans=topic_data.get("evidence", []),
                relevance_score=float(topic_data.get("relevance", 0.5)),
            ))

        # Parse investor intent
        intent_data = enrichment.get("investor_intent", {})
        investor_intent = InvestorIntent(
            primary_intent=intent_data.get("intent", "exploration"),
            confidence=float(intent_data.get("confidence", 0.5)),
            reasoning=intent_data.get("reasoning", ""),
        ) if intent_data else None

        # Parse response posture
        posture_data = enrichment.get("response_posture", {})
        response_posture = ResponsePosture(
            primary_posture=posture_data.get("posture", "neutral"),
            confidence=float(posture_data.get("confidence", 0.5)),
            reasoning=posture_data.get("reasoning", ""),
        ) if posture_data else None

        # Parse evidence spans
        evidence = []
        for ev_data in enrichment.get("key_evidence", []):
            evidence.append(EvidenceSpan(
                quote=ev_data.get("quote", ""),
                page_number=int(ev_data.get("page", qa.start_page)),
                speaker_name=ev_data.get("speaker"),
                relevance=ev_data.get("relevance", ""),
            ))

        # Get summary
        summary = enrichment.get("summary", "")

        return EnrichedQAUnit(
            **base_fields,
            topics=topics,
            investor_intent=investor_intent,
            response_posture=response_posture,
            key_evidence=evidence,
            summary=summary,
        )

    except Exception as e:
        logger.error("qa_enrichment_failed", qa_id=qa.qa_id, error=str(e))
        # Return with minimal enrichment
        return EnrichedQAUnit(
            **base_fields,
            topics=[],
            investor_intent=None,
            response_posture=None,
            key_evidence=[],
            summary=None,
        )




def _enrich_single_strategic(stmt: ExtractedStrategicStatement) -> EnrichedStrategicStatement:
    """Enrich a single strategic statement with LLM analysis."""

    content = f"""Strategic Statement ({stmt.speaker_name}):
{stmt.text}

Type: {stmt.statement_type}
Forward-looking: {stmt.is_forward_looking}"""

    try:
        enrichment = run_enrichment_chain(
            unit_type="Strategic Statement",
            content=content,
            start_page=stmt.page_number,
            end_page=stmt.page_number,
        )

        # Parse topics
        topics = []
        for topic_data in enrichment.get("topics", []):
            topics.append(TopicMention(
                topic_name=topic_data.get("name", "Unknown"),
                topic_category=topic_data.get("category", "Other"),
                evidence_spans=topic_data.get("evidence", []),
                relevance_score=float(topic_data.get("relevance", 0.5)),
            ))

        # Parse evidence spans
        evidence = []
        for ev_data in enrichment.get("key_evidence", []):
            evidence.append(EvidenceSpan(
                quote=ev_data.get("quote", ""),
                page_number=int(ev_data.get("page", stmt.page_number)),
                speaker_name=ev_data.get("speaker", stmt.speaker_name),
                relevance=ev_data.get("relevance", ""),
            ))

        # Get summary
        summary = enrichment.get("summary", "")

        return EnrichedStrategicStatement(
            # Copy base fields
            statement_id=stmt.statement_id,
            source_section_id=stmt.source_section_id,
            speaker_name=stmt.speaker_name,
            speaker_id=stmt.speaker_id,
            text=stmt.text,
            statement_type=stmt.statement_type,
            is_forward_looking=stmt.is_forward_looking,
            page_number=stmt.page_number,
            sequence_in_section=stmt.sequence_in_section,
            # Enrichment
            topics=topics,
            key_evidence=evidence,
            summary=summary,
        )

    except Exception as e:
        logger.error("strategic_enrichment_failed", statement_id=stmt.statement_id, error=str(e))
        # Return with minimal enrichment
        return EnrichedStrategicStatement(
            statement_id=stmt.statement_id,
            source_section_id=stmt.source_section_id,
            speaker_name=stmt.speaker_name,
            speaker_id=stmt.speaker_id,
            text=stmt.text,
            statement_type=stmt.statement_type,
            is_forward_looking=stmt.is_forward_looking,
            page_number=stmt.page_number,
            sequence_in_section=stmt.sequence_in_section,
            topics=[],
            key_evidence=[],
            summary=None,
        )


# =============================================================================
# Main Enrichment Functions
# =============================================================================

def enrich_qa_units(
    qa_result: QAExtractionResult,
    max_units: Optional[int] = None,
) -> list[EnrichedQAUnit]:
    """Enrich all Q&A units with topics, intent, posture, and evidence.

    Args:
        qa_result: Result from Q&A extraction stage.
        max_units: Optional limit on units to enrich (for testing/cost control).

    Returns:
        List of EnrichedQAUnit objects.
    """
    logger.info("qa_enrichment_start", total_units=qa_result.total_qa_units)

    enriched_units = []
    units_to_process = qa_result.qa_units[:max_units] if max_units else qa_result.qa_units

    for i, qa in enumerate(units_to_process):
        logger.debug("enriching_qa", qa_id=qa.qa_id, index=i, total=len(units_to_process))

        enriched = _enrich_single_qa(qa)
        enriched_units.append(enriched)

        if (i + 1) % 5 == 0:
            logger.info("qa_enrichment_progress", completed=i + 1, total=len(units_to_process))

    logger.info(
        "qa_enrichment_complete",
        enriched_count=len(enriched_units),
        with_topics=sum(1 for e in enriched_units if e.topics),
        with_intent=sum(1 for e in enriched_units if e.investor_intent),
    )

    return enriched_units


def enrich_strategic_statements(
    strategic_result: StrategicExtractionResult,
    max_statements: Optional[int] = None,
) -> list[EnrichedStrategicStatement]:
    """Enrich all strategic statements with topics and evidence.

    Args:
        strategic_result: Result from strategic extraction stage.
        max_statements: Optional limit on statements to enrich.

    Returns:
        List of EnrichedStrategicStatement objects.
    """
    logger.info("strategic_enrichment_start", total_statements=strategic_result.total_statements)

    enriched_statements = []
    statements_to_process = (
        strategic_result.statements[:max_statements]
        if max_statements
        else strategic_result.statements
    )

    for i, stmt in enumerate(statements_to_process):
        logger.debug("enriching_strategic", statement_id=stmt.statement_id, index=i)

        enriched = _enrich_single_strategic(stmt)
        enriched_statements.append(enriched)

        if (i + 1) % 5 == 0:
            logger.info(
                "strategic_enrichment_progress",
                completed=i + 1,
                total=len(statements_to_process),
            )

    logger.info(
        "strategic_enrichment_complete",
        enriched_count=len(enriched_statements),
        with_topics=sum(1 for e in enriched_statements if e.topics),
    )

    return enriched_statements
