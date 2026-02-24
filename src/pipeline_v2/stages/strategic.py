"""Stage 5: Strategic Statement Extraction - Extract strategic statements from opening/closing.

Responsibilities:
- Extract guidance, outlook, and strategic initiatives from prepared remarks
- Classify statement types
- Identify forward-looking statements
- Link to speaker registry
"""

import json
import re
from typing import Optional

import structlog

from src.llm.chains import run_strategic_extraction_chain
from src.pipeline_v2.models import (
    BoundaryDetectionResult,
    ExtractedStrategicStatement,
    SectionType,
    SpeakerRegistry,
    StrategicExtractionResult,
    TranscriptSection,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Statement Type Patterns
# =============================================================================

# Forward-looking language patterns
FORWARD_LOOKING_PATTERNS = [
    r"(?i)we\s+(?:expect|anticipate|believe|plan|intend|project|forecast)",
    r"(?i)(?:looking|moving)\s+(?:forward|ahead)",
    r"(?i)(?:in|for)\s+(?:the\s+)?(?:coming|next|future)",
    r"(?i)(?:our|the)\s+(?:outlook|guidance|target|goal)",
    r"(?i)(?:will|should|may|could)\s+(?:be|see|experience|achieve|deliver|grow)",
    r"(?i)(?:by|through|over)\s+(?:the\s+)?(?:end\s+of\s+)?(?:FY|fiscal|Q[1-4]|\d{4})",
]

# Guidance patterns
GUIDANCE_PATTERNS = [
    r"(?i)(?:raise|maintain|lower|update|revise|reiterate)\s+(?:our\s+)?(?:guidance|outlook|forecast)",
    r"(?i)(?:expect|guide|target)(?:ing)?\s+(?:revenue|earnings|EPS|margin|growth)",
    r"(?i)(?:full[- ]?year|annual|quarterly)\s+(?:guidance|outlook|expectations?)",
    r"(?i)(?:eps|earnings\s+per\s+share)\s+(?:of|between|around)",
]

# Strategic initiative patterns
STRATEGIC_PATTERNS = [
    r"(?i)(?:strategic|key)\s+(?:initiative|priority|focus|investment)",
    r"(?i)(?:invest(?:ing)?|expand(?:ing)?|launch(?:ing)?|develop(?:ing)?)\s+(?:in|our|new)",
    r"(?i)(?:transformation|transition|restructuring|optimization)",
    r"(?i)(?:market\s+)?(?:expansion|penetration|share\s+gains?)",
]

# Operational update patterns
OPERATIONAL_PATTERNS = [
    r"(?i)(?:achieved|delivered|completed|executed|implemented)",
    r"(?i)(?:operational|operating)\s+(?:performance|efficiency|improvement)",
    r"(?i)(?:cost\s+)?(?:reduction|savings|efficiency)",
]

# Financial highlight patterns
FINANCIAL_PATTERNS = [
    r"(?i)(?:revenue|sales|earnings|income|margin|growth)\s+(?:of|was|were|increased|decreased|grew)",
    r"(?i)(?:year[- ]over[- ]year|quarter[- ]over[- ]quarter|sequential)\s+(?:growth|increase|decline)",
    r"(?i)(?:record|strong|solid|robust)\s+(?:revenue|earnings|results|performance)",
]

# Risk disclosure patterns
RISK_PATTERNS = [
    r"(?i)(?:risk|challenge|headwind|uncertainty|pressure)",
    r"(?i)(?:macro|macroeconomic|economic\s+)?(?:environment|conditions|volatility)",
    r"(?i)(?:supply\s+chain|inflation|fx|foreign\s+exchange)\s+(?:impact|pressure|headwind)",
]


# =============================================================================
# Helper Functions
# =============================================================================

def _is_forward_looking(text: str) -> bool:
    """Check if statement contains forward-looking language."""
    for pattern in FORWARD_LOOKING_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _classify_statement_type(text: str) -> str:
    """Classify the type of strategic statement."""
    text_lower = text.lower()

    # Check patterns in order of specificity
    for pattern in GUIDANCE_PATTERNS:
        if re.search(pattern, text):
            return "guidance"

    # Check for outlook (more general than guidance)
    if "outlook" in text_lower or re.search(r"(?i)expect.*(?:year|quarter)", text):
        return "outlook"

    for pattern in STRATEGIC_PATTERNS:
        if re.search(pattern, text):
            return "strategic_initiative"

    for pattern in OPERATIONAL_PATTERNS:
        if re.search(pattern, text):
            return "operational_update"

    for pattern in FINANCIAL_PATTERNS:
        if re.search(pattern, text):
            return "financial_highlight"

    for pattern in RISK_PATTERNS:
        if re.search(pattern, text):
            return "risk_disclosure"

    return "other"


def _extract_statements_deterministic(
    text: str,
    section_id: str,
    start_page: int,
    speakers: list[str],
) -> list[ExtractedStrategicStatement]:
    """Extract strategic statements using pattern matching.

    Simple approach: split into sentences and classify each.
    """
    statements = []
    statement_counter = 0

    # Split into paragraphs, then sentences
    paragraphs = re.split(r"\n\s*\n", text)

    current_speaker = speakers[0] if speakers else "Management"

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 50:  # Skip very short paragraphs
            continue

        # Check if this paragraph starts with a speaker name
        speaker_match = re.match(r"^([A-Z][A-Za-z\s\.\-']+?)[-–—:]\s*(.+)", para, re.DOTALL)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            para = speaker_match.group(2).strip()

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:  # Skip very short sentences
                continue

            statement_type = _classify_statement_type(sentence)

            # Only extract meaningful strategic statements
            if statement_type != "other":
                is_forward = _is_forward_looking(sentence)

                statements.append(ExtractedStrategicStatement(
                    statement_id=f"stmt_{statement_counter:03d}",
                    source_section_id=section_id,
                    speaker_name=current_speaker,
                    speaker_id=None,
                    text=sentence,
                    statement_type=statement_type,
                    is_forward_looking=is_forward,
                    page_number=start_page,
                    sequence_in_section=statement_counter,
                ))
                statement_counter += 1

    return statements


def _extract_statements_with_llm(
    section: TranscriptSection,
    registry: Optional[SpeakerRegistry],
) -> list[ExtractedStrategicStatement]:
    """Use LLM to extract strategic statements."""
    logger.debug("llm_strategic_extraction_start", section_id=section.section_id)

    # Prepare turn data
    turns_data = []
    for speaker in section.detected_speakers:
        turns_data.append({
            "speaker": speaker,
            "text": section.raw_text[:2000],  # Truncate for context
        })

    if not turns_data:
        turns_data = [{"speaker": "Management", "text": section.raw_text[:2000]}]

    try:
        llm_result = run_strategic_extraction_chain(json.dumps(turns_data))
        logger.debug("llm_strategic_result", statements=len(llm_result.get("strategic_statements", [])))

        statements = []
        for i, stmt_data in enumerate(llm_result.get("strategic_statements", [])):
            statements.append(ExtractedStrategicStatement(
                statement_id=f"stmt_{i:03d}",
                source_section_id=section.section_id,
                speaker_name=stmt_data.get("speaker", "Management"),
                speaker_id=None,
                text=stmt_data.get("text", ""),
                statement_type=stmt_data.get("type", "other"),
                is_forward_looking=stmt_data.get("forward_looking", False),
                page_number=section.start_page,
                sequence_in_section=i,
            ))

        return statements

    except Exception as e:
        logger.error("llm_strategic_extraction_failed", error=str(e))
        return []


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_strategic_statements(
    boundary_result: BoundaryDetectionResult,
    registry: Optional[SpeakerRegistry] = None,
    use_llm: bool = True,
) -> StrategicExtractionResult:
    """Extract strategic statements from opening and closing sections.

    Args:
        boundary_result: Result from boundary detection.
        registry: Optional speaker registry for ID resolution.
        use_llm: Whether to use LLM for extraction.

    Returns:
        StrategicExtractionResult with all extracted statements.
    """
    logger.info("strategic_extraction_start", total_sections=len(boundary_result.sections))

    all_statements: list[ExtractedStrategicStatement] = []
    sections_processed = 0

    # Filter to opening and closing sections
    relevant_sections = [
        s for s in boundary_result.sections
        if s.section_type in (SectionType.OPENING_REMARKS, SectionType.CLOSING_REMARKS)
    ]

    logger.debug("relevant_sections_found", count=len(relevant_sections))

    for section in relevant_sections:
        sections_processed += 1

        if use_llm:
            # Use LLM for better extraction
            statements = _extract_statements_with_llm(section, registry)
        else:
            # Deterministic extraction
            statements = _extract_statements_deterministic(
                text=section.raw_text,
                section_id=section.section_id,
                start_page=section.start_page,
                speakers=section.detected_speakers,
            )

        if not statements and not use_llm:
            # Fallback to LLM if deterministic found nothing
            statements = _extract_statements_with_llm(section, registry)

        if statements:
            # Renumber to be globally unique
            base_id = len(all_statements)
            for i, stmt in enumerate(statements):
                stmt.statement_id = f"stmt_{base_id + i:03d}"
                stmt.sequence_in_section = base_id + i

                # Resolve speaker ID
                if registry:
                    speaker_info = registry.get_by_name(stmt.speaker_name)
                    if speaker_info:
                        stmt.speaker_id = speaker_info.speaker_id

            all_statements.extend(statements)

        logger.debug(
            "section_processed",
            section_id=section.section_id,
            statements_found=len(statements),
        )

    # Compute statistics
    forward_looking_count = sum(1 for s in all_statements if s.is_forward_looking)

    result = StrategicExtractionResult(
        statements=all_statements,
        total_statements=len(all_statements),
        forward_looking_count=forward_looking_count,
        sections_processed=sections_processed,
    )

    logger.info(
        "strategic_extraction_complete",
        total_statements=len(all_statements),
        forward_looking=forward_looking_count,
        sections_processed=sections_processed,
    )

    return result
