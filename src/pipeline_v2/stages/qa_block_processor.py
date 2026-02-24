"""Stage 4B: Q&A Block Processor - Block-by-Block LLM Structuring.

HYBRID APPROACH:
- Regex detects Q&A BLOCKS (investor introductions)
- LLM structures EACH BLOCK internally
- LLM MUST return a result for every block (never silent drop)

PHILOSOPHY:
- Regex proposes blocks (high recall)
- LLM labels and structures (semantic understanding)
- Nothing is silently dropped

BLOCK PROCESSING:
- Each block = one investor's complete interaction
- LLM determines question/answer boundaries within the block
- LLM detects follow-up questions
- LLM handles multi-speaker answers
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Literal

import structlog

from src.pipeline_v2.stages.boundary import QABlock, detect_qa_blocks
from src.pipeline_v2.models import (
    ExtractedQAUnit,
    QAExtractionResult,
    SpeakerRegistry,
    SpeakerRole,
    SpeakerTurn,
    TranscriptSection,
    SectionType,
)
from src.pipeline_v2.llm_helpers import _invoke_llm

logger = structlog.get_logger(__name__)


# =============================================================================
# Trace Structures
# =============================================================================

@dataclass
class BlockProcessingResult:
    """Result of processing a single Q&A block."""
    block_id: str
    is_valid: bool
    reason: Optional[str]  # Why invalid (if is_valid=False)
    qa_pairs: list[ExtractedQAUnit]
    questioner_name: str
    questioner_company: Optional[str]
    llm_confidence: float
    llm_reasoning: str


@dataclass
class QABlockExtractionTrace:
    """Trace of block-by-block Q&A extraction."""
    blocks_detected: int = 0
    blocks_processed: int = 0
    blocks_valid: int = 0
    blocks_invalid: int = 0
    invalid_reasons: list[dict] = field(default_factory=list)
    total_qa_pairs: int = 0
    llm_calls: int = 0


# =============================================================================
# Speaker Turn Extraction (Regex)
# =============================================================================

SPEAKER_LABEL_PATTERN = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){0,3})\s*[-–—:]\s*(?P<text>.*)$",
    re.MULTILINE
)


def _extract_speaker_turns(text: str) -> list[dict]:
    """Extract speaker turns from text block.

    Returns list of {speaker, text, line_num} dicts.
    """
    lines = text.split('\n')
    turns = []
    current_speaker = None
    current_text = []
    current_line = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check for speaker label
        match = SPEAKER_LABEL_PATTERN.match(line)
        if match:
            # Save previous turn
            if current_speaker and current_text:
                turns.append({
                    "speaker": current_speaker,
                    "text": ' '.join(current_text),
                    "line_num": current_line,
                })

            current_speaker = match.group("speaker").strip()
            current_text = [match.group("text").strip()] if match.group("text").strip() else []
            current_line = i
        elif current_speaker:
            # Continue current speaker's turn
            current_text.append(line)

    # Save last turn
    if current_speaker and current_text:
        turns.append({
            "speaker": current_speaker,
            "text": ' '.join(current_text),
            "line_num": current_line,
        })

    return turns


# =============================================================================
# LLM Block Structuring
# =============================================================================

def _build_registry_context(registry: Optional[SpeakerRegistry]) -> str:
    """Build speaker registry context for LLM prompt."""
    if not registry or not registry.speakers:
        return "No speaker registry available."

    lines = []
    for speaker_id, info in registry.speakers.items():
        role = info.role.value if info.role else "unknown"
        title = f" ({info.title})" if info.title else ""
        lines.append(f"- {info.canonical_name}{title}: {role}")

    return "SPEAKER REGISTRY (authoritative):\n" + "\n".join(lines[:15])


def _structure_qa_block_with_llm(
    block: QABlock,
    speaker_turns: list[dict],
    registry: Optional[SpeakerRegistry],
) -> BlockProcessingResult:
    """Use LLM to VALIDATE and identify BOUNDARIES in a Q&A block.

    CRITICAL: LLM returns TURN INDICES only - code extracts FULL text.
    This prevents LLM from truncating content.

    Args:
        block: The regex-detected Q&A block
        speaker_turns: Extracted speaker turns within the block (with FULL text)
        registry: Global speaker registry for role validation

    Returns:
        BlockProcessingResult with structured Q&A pairs (full text preserved)
    """
    registry_context = _build_registry_context(registry)

    # Build turn summary - show enough for LLM to understand, but text will come from original
    turn_summary = ""
    for i, turn in enumerate(speaker_turns[:30]):
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        # Show first 200 chars + indicator if truncated (LLM uses indices, code uses full text)
        preview = text[:200] + "..." if len(text) > 200 else text
        turn_summary += f"[Turn {i}] {speaker}: {preview}\n"

    prompt = f"""You are VALIDATING Q&A boundaries in an earnings call transcript block.

CRITICAL: Return TURN INDICES only. DO NOT return the actual text content.
The system will extract FULL text using your indices.

{registry_context}

=== BLOCK INFO ===
Introduced Questioner: {block.questioner_name}
Company: {block.questioner_company or "Unknown"}
Pages: {block.start_page}-{block.end_page}

=== SPEAKER TURNS (with indices) ===
{turn_summary}

=== YOUR TASK ===

1. VALIDATE: Is this a valid Q&A block?
   - Valid: Contains at least one question from an analyst and one answer from management
   - Invalid: Opening remarks, moderator announcements, greetings only

2. IDENTIFY BOUNDARIES: If valid, return the TURN INDICES for each Q&A pair:
   - question_turn_indices: Which turn(s) contain the question
   - answer_turn_indices: Which turn(s) contain the answer
   - Multiple consecutive turns can form one question or answer

=== RULES ===
- Moderator turns (introductions) should NOT be included in question/answer indices
- Questions from analysts: include ALL their consecutive turns before management responds
- Answers from management: include ALL consecutive management turns until next analyst speaks
- Follow-ups: same analyst continues after getting an answer

=== OUTPUT FORMAT ===
Return a JSON object:
{{
  "is_valid": true/false,
  "reason": "Why invalid (only if is_valid=false)",
  "qa_pairs": [
    {{
      "question_turn_indices": [0, 1],
      "answer_turn_indices": [2, 3, 4],
      "question_speaker": "Analyst Name",
      "answer_speakers": ["Manager1", "Manager2"],
      "is_follow_up": false,
      "follow_up_of_index": null,
      "topics": ["topic1", "topic2"]
    }}
  ],
  "confidence": 0.9,
  "reasoning": "Brief explanation"
}}

IMPORTANT:
- Output ONLY valid JSON
- Return INDICES, not text content
- Indices are 0-based turn numbers from the list above"""

    response = _invoke_llm(prompt, max_tokens=1500)
    return _parse_block_result(response, block, speaker_turns)


def _parse_block_result(
    response: str,
    block: QABlock,
    speaker_turns: list[dict],
) -> BlockProcessingResult:
    """Parse LLM response and extract FULL text using turn indices.

    CRITICAL: LLM returns indices, we extract FULL text from original speaker_turns.
    This prevents text truncation by LLM.
    """
    parse_error = "No JSON found in response"

    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())

            is_valid = data.get("is_valid", False)
            reason = data.get("reason")
            confidence = data.get("confidence", 0.5)
            reasoning = data.get("reasoning", "")

            qa_pairs = []
            for i, qa in enumerate(data.get("qa_pairs", [])):
                # Extract FULL question text from original speaker_turns using indices
                question_indices = qa.get("question_turn_indices", [])
                question_texts = []
                question_turns_list = []

                for idx in question_indices:
                    if 0 <= idx < len(speaker_turns):
                        turn = speaker_turns[idx]
                        full_text = turn.get("text", "")  # FULL text, not truncated
                        question_texts.append(full_text)
                        question_turns_list.append(SpeakerTurn(
                            speaker_name=turn.get("speaker", block.questioner_name),
                            speaker_id=None,
                            text=full_text,
                            page_number=block.start_page,
                            is_question=True,
                        ))

                # Extract FULL answer text from original speaker_turns using indices
                answer_indices = qa.get("answer_turn_indices", [])
                answer_texts = []
                response_turns_list = []
                answer_speakers_set = set()

                for idx in answer_indices:
                    if 0 <= idx < len(speaker_turns):
                        turn = speaker_turns[idx]
                        full_text = turn.get("text", "")  # FULL text, not truncated
                        speaker_name = turn.get("speaker", "Management")
                        answer_texts.append(full_text)
                        answer_speakers_set.add(speaker_name)
                        response_turns_list.append(SpeakerTurn(
                            speaker_name=speaker_name,
                            speaker_id=None,
                            text=full_text,
                            page_number=block.end_page,
                            is_question=False,
                        ))

                # Combine texts with proper spacing
                full_question_text = " ".join(question_texts)
                full_answer_text = " ".join(answer_texts)
                answer_speakers = list(answer_speakers_set) or qa.get("answer_speakers", ["Management"])

                # Use LLM-provided speaker names as fallback
                question_speaker = qa.get("question_speaker", block.questioner_name)

                qa_id = f"{block.block_id}_qa_{i:02d}"
                follow_up_of = None
                if qa.get("is_follow_up") and qa.get("follow_up_of_index") is not None:
                    follow_up_of = f"{block.block_id}_qa_{qa.get('follow_up_of_index'):02d}"

                qa_pairs.append(ExtractedQAUnit(
                    qa_id=qa_id,
                    source_section_id=block.block_id,
                    questioner_name=question_speaker,
                    questioner_id=None,
                    question_turns=question_turns_list if question_turns_list else [SpeakerTurn(
                        speaker_name=question_speaker,
                        speaker_id=None,
                        text=full_question_text,
                        page_number=block.start_page,
                        is_question=True,
                    )],
                    response_turns=response_turns_list if response_turns_list else [SpeakerTurn(
                        speaker_name=answer_speakers[0] if answer_speakers else "Management",
                        speaker_id=None,
                        text=full_answer_text,
                        page_number=block.end_page,
                        is_question=False,
                    )],
                    responder_names=answer_speakers,
                    responder_ids=[],
                    question_text=full_question_text,
                    response_text=full_answer_text,
                    is_follow_up=qa.get("is_follow_up", False),
                    follow_up_of=follow_up_of,
                    start_page=block.start_page,
                    end_page=block.end_page,
                    sequence_in_session=i,
                    boundary_justification=reasoning[:500],
                ))

            return BlockProcessingResult(
                block_id=block.block_id,
                is_valid=is_valid,
                reason=reason,
                qa_pairs=qa_pairs,
                questioner_name=block.questioner_name,
                questioner_company=block.questioner_company,
                llm_confidence=confidence,
                llm_reasoning=reasoning,
            )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        parse_error = str(e)
        logger.warning("block_json_parse_failed", block_id=block.block_id, error=parse_error)

    # Fallback: return invalid result (but don't drop the block)
    return BlockProcessingResult(
        block_id=block.block_id,
        is_valid=False,
        reason=f"LLM response parse failed: {parse_error[:100]}",
        qa_pairs=[],
        questioner_name=block.questioner_name,
        questioner_company=block.questioner_company,
        llm_confidence=0.0,
        llm_reasoning="Parse failure",
    )


# =============================================================================
# Main Processing Function
# =============================================================================

def extract_qa_from_blocks(
    qa_section: TranscriptSection,
    full_text: str,
    total_pages: int,
    registry: Optional[SpeakerRegistry] = None,
) -> tuple[QAExtractionResult, QABlockExtractionTrace]:
    """Extract Q&A units from a Q&A section using block-by-block processing.

    APPROACH:
    1. Regex detects Q&A blocks (investor introductions)
    2. LLM structures each block internally
    3. LLM MUST return a result for every block

    Args:
        qa_section: The Q&A section from boundary detection
        full_text: Full document text (for page estimation)
        total_pages: Total pages in document
        registry: Global speaker registry

    Returns:
        Tuple of (QAExtractionResult, QABlockExtractionTrace)
    """
    trace = QABlockExtractionTrace()

    logger.info(
        "qa_block_extraction_start",
        section_id=qa_section.section_id,
        text_length=len(qa_section.raw_text),
    )

    # Step 1: Detect Q&A blocks using regex
    blocks = detect_qa_blocks(
        qa_section_text=qa_section.raw_text,
        section_start_offset=qa_section.char_offset_start,
        section_start_page=qa_section.start_page,
        section_end_page=qa_section.end_page,
        text_length=len(full_text),
        total_pages=total_pages,
    )

    trace.blocks_detected = len(blocks)
    logger.info("qa_blocks_detected", count=len(blocks))

    # Step 2: Process each block with LLM
    all_qa_units = []

    for block in blocks:
        trace.blocks_processed += 1
        trace.llm_calls += 1

        # Extract speaker turns from block
        speaker_turns = _extract_speaker_turns(block.raw_text)

        logger.debug(
            "processing_block",
            block_id=block.block_id,
            questioner=block.questioner_name,
            turns=len(speaker_turns),
        )

        # Structure block with LLM
        result = _structure_qa_block_with_llm(block, speaker_turns, registry)

        if result.is_valid:
            trace.blocks_valid += 1
            all_qa_units.extend(result.qa_pairs)
            trace.total_qa_pairs += len(result.qa_pairs)
            logger.debug(
                "block_valid",
                block_id=block.block_id,
                qa_pairs=len(result.qa_pairs),
            )
        else:
            trace.blocks_invalid += 1
            trace.invalid_reasons.append({
                "block_id": block.block_id,
                "questioner": block.questioner_name,
                "reason": result.reason,
            })
            logger.debug(
                "block_invalid",
                block_id=block.block_id,
                reason=result.reason,
            )

    # Resolve speaker IDs from registry
    if registry:
        for qa in all_qa_units:
            if qa.questioner_name:
                info = registry.get_by_name(qa.questioner_name)
                if info:
                    qa.questioner_id = info.speaker_id

            resolved_ids = []
            for name in qa.responder_names:
                info = registry.get_by_name(name)
                if info:
                    resolved_ids.append(info.speaker_id)
            qa.responder_ids = resolved_ids

    # Renumber Q&A units globally
    for i, qa in enumerate(all_qa_units):
        qa.qa_id = f"qa_{i:03d}"
        qa.sequence_in_session = i

    # Compute statistics
    total_follow_ups = sum(1 for qa in all_qa_units if qa.is_follow_up)
    unique_questioners = len(set(qa.questioner_name for qa in all_qa_units))

    result = QAExtractionResult(
        qa_units=all_qa_units,
        total_qa_units=len(all_qa_units),
        total_follow_ups=total_follow_ups,
        unique_questioners=unique_questioners,
        sections_processed=1,
        sections_with_no_qa=[] if all_qa_units else [qa_section.section_id],
    )

    logger.info(
        "qa_block_extraction_complete",
        blocks_detected=trace.blocks_detected,
        blocks_valid=trace.blocks_valid,
        blocks_invalid=trace.blocks_invalid,
        total_qa_pairs=trace.total_qa_pairs,
        follow_ups=total_follow_ups,
    )

    return result, trace


def process_all_qa_sections(
    boundary_result,
    full_text: str,
    total_pages: int,
    registry: Optional[SpeakerRegistry] = None,
) -> tuple[QAExtractionResult, QABlockExtractionTrace]:
    """Process all Q&A sections in the document.

    Args:
        boundary_result: Result from boundary detection
        full_text: Full document text
        total_pages: Total pages
        registry: Speaker registry

    Returns:
        Combined QAExtractionResult and trace
    """
    all_qa_units = []
    combined_trace = QABlockExtractionTrace()

    qa_sections = [s for s in boundary_result.sections if s.section_type == SectionType.QA_SESSION]

    for section in qa_sections:
        result, trace = extract_qa_from_blocks(
            qa_section=section,
            full_text=full_text,
            total_pages=total_pages,
            registry=registry,
        )

        all_qa_units.extend(result.qa_units)
        combined_trace.blocks_detected += trace.blocks_detected
        combined_trace.blocks_processed += trace.blocks_processed
        combined_trace.blocks_valid += trace.blocks_valid
        combined_trace.blocks_invalid += trace.blocks_invalid
        combined_trace.invalid_reasons.extend(trace.invalid_reasons)
        combined_trace.total_qa_pairs += trace.total_qa_pairs
        combined_trace.llm_calls += trace.llm_calls

    # Renumber all Q&A units
    for i, qa in enumerate(all_qa_units):
        qa.qa_id = f"qa_{i:03d}"
        qa.sequence_in_session = i

    total_follow_ups = sum(1 for qa in all_qa_units if qa.is_follow_up)
    unique_questioners = len(set(qa.questioner_name for qa in all_qa_units))

    final_result = QAExtractionResult(
        qa_units=all_qa_units,
        total_qa_units=len(all_qa_units),
        total_follow_ups=total_follow_ups,
        unique_questioners=unique_questioners,
        sections_processed=len(qa_sections),
        sections_with_no_qa=[],
    )

    return final_result, combined_trace
