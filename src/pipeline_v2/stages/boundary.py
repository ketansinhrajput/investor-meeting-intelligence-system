"""Stage 2: Boundary Detection - Hybrid Intelligence Approach.

Philosophy: Deterministic structure + LLM confirmation for ambiguity.

Approach:
1. Deterministic patterns for HIGH-RECALL candidate detection
2. State machine for section transitions
3. LLM confirmation for ambiguous boundaries
4. Inspectable intermediate outputs
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.pipeline_v2.models import (
    BoundaryDetectionResult,
    DetectionSignal,
    SectionType,
    TranscriptSection,
)
from src.pipeline_v2.llm_helpers import confirm_qa_session_start, confirm_section_boundary

logger = structlog.get_logger(__name__)


# =============================================================================
# Inspectable Intermediate Structures
# =============================================================================

@dataclass
class BoundaryCandidate:
    """A candidate boundary detected by patterns."""
    char_offset: int
    text_snippet: str
    signal_type: DetectionSignal
    pattern_matched: str
    llm_confirmed: Optional[bool] = None
    llm_confidence: float = 0.0
    llm_reasoning: str = ""


@dataclass
class BoundaryDetectionTrace:
    """Inspectable trace of boundary detection process."""
    candidates: list[BoundaryCandidate] = field(default_factory=list)
    state_transitions: list[dict] = field(default_factory=list)
    final_boundaries: list[int] = field(default_factory=list)


# =============================================================================
# Pattern Definitions (High-Recall Candidate Detection)
# =============================================================================

# =============================================================================
# EXPLICIT Q&A SESSION START (HIGH PRIORITY)
# These are clear moderator phrases that indicate Q&A is starting
# =============================================================================
EXPLICIT_QA_START_PATTERNS = [
    # "We will now begin the question-and-answer session"
    (r"(?i)we\s+will\s+now\s+begin\s+the\s+question[\s-]*(?:and)?[\s-]*answer", "explicit_qa_begin"),
    # "We'll now open the floor for questions"
    (r"(?i)(?:we(?:'ll)?|let(?:'s)?)\s+(?:now\s+)?open\s+the\s+floor\s+for\s+questions?", "open_floor"),
    # "We will now take questions"
    (r"(?i)we\s+will\s+now\s+(?:take|accept)\s+questions?", "take_questions"),
    # "Let's begin the Q&A"
    (r"(?i)let(?:'s)?\s+begin\s+(?:the\s+)?(?:Q\s*&?\s*A|question)", "lets_begin_qa"),
]

# Q&A session start candidates (loose patterns for high recall - fallback)
QA_START_CANDIDATES = [
    (r"(?i)question(?:s)?\s*(?:and|&)?\s*answer", "qa_header"),
    (r"(?i)(?:we(?:'ll)?|let(?:'s)?|now)\s+(?:open|begin|start|take)", "qa_open"),
    (r"(?i)(?:first|next)\s+question\s+(?:is\s+)?(?:from|comes)", "first_question"),
    (r"(?i)(?:floor|line(?:s)?)\s+(?:is\s+)?(?:open|now)", "floor_open"),
    (r"(?i)operator[,:]?\s+(?:we(?:'re)?|please)", "operator_cue"),
    (r"(?i)(?:go\s+ahead|proceed)\s+with\s+(?:your\s+)?question", "go_ahead"),
]

# =============================================================================
# Q&A BLOCK DETECTION (Investor Introduction Patterns)
# Each block starts when moderator introduces a new questioner
# =============================================================================
QA_BLOCK_START_PATTERNS = [
    # "The first question is from the line of Amandeep Singh from Ambit Capital"
    (r"(?i)(?:the\s+)?(?:first|next|following)\s+question\s+(?:is\s+)?(?:from|comes\s+from)\s+(?:the\s+line\s+of\s+)?([A-Z][A-Za-z\.\-'\s]+?)(?:\s+from\s+([A-Za-z\s&,\.]+?))?(?:\.|$)", "question_from_line"),
    # "We have a question from John Smith"
    (r"(?i)we\s+(?:have|got)\s+(?:a\s+)?question\s+from\s+([A-Z][A-Za-z\.\-'\s]+?)(?:\s+(?:from|of|with)\s+([A-Za-z\s&,\.]+?))?(?:\.|$)", "we_have_question"),
    # "Our next participant is..."
    (r"(?i)(?:our\s+)?next\s+(?:question|participant|caller)\s+(?:is|comes)\s+(?:from\s+)?([A-Z][A-Za-z\.\-'\s]+?)(?:\.|$)", "next_participant"),
    # "Please go ahead, [Name]"
    (r"(?i)please\s+go\s+ahead[,\s]+([A-Z][A-Za-z\.\-'\s]+?)(?:\.|$)", "go_ahead_name"),
    # "[Name], you may ask your question"
    (r"(?i)([A-Z][A-Za-z\.\-'\s]+?)[,\s]+(?:you\s+may|please)\s+(?:ask|go\s+ahead|proceed)", "name_may_ask"),
]

# Closing section candidates
CLOSING_CANDIDATES = [
    (r"(?i)(?:that|this)\s+(?:concludes?|ends?|wraps?\s+up)", "concludes"),
    (r"(?i)(?:thank\s+you|thanks)\s+(?:all\s+)?(?:for\s+)?(?:joining|participating)", "thanks"),
    (r"(?i)(?:no\s+(?:more|further)\s+questions?)", "no_more_questions"),
    (r"(?i)(?:this|that)\s+(?:will\s+)?(?:be|is)\s+all\s+for\s+today", "all_for_today"),
]

# Moderator cue patterns (within Q&A)
MODERATOR_CUE_PATTERNS = [
    (r"(?i)(?:next|our\s+next)\s+question\s+(?:is\s+)?(?:from|comes)", "next_question"),
    (r"(?i)(?:we(?:'ll)?|let(?:'s)?)\s+(?:go\s+to|take|move\s+to)\s+(?:the\s+)?next", "move_next"),
    (r"(?i)(?:from\s+the\s+line\s+of)", "from_line_of"),
]

# Speaker label pattern (strict - 1-4 words)
SPEAKER_LABEL_PATTERN = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){0,3})\s*[-–—:]\s*",
    re.MULTILINE
)

# Content indicator words (for rejecting invalid speakers)
CONTENT_INDICATOR_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'that', 'which',
    'who', 'whom', 'this', 'these', 'those', 'what', 'conference', 'call',
    'reminder', 'hand', 'over', 'begin', 'thank', 'thanks', 'you', 'we', 'i',
    'they', 'he', 'she', 'now', 'before', 'after', 'there', 'here',
}


# =============================================================================
# State Machine for Section Transitions
# =============================================================================

class SectionStateMachine:
    """Deterministic state machine for section transitions."""

    def __init__(self):
        self.current_state = SectionType.OPENING_REMARKS
        self.transitions: list[dict] = []

    def can_transition_to(self, new_state: SectionType) -> bool:
        """Check if transition is valid."""
        valid_transitions = {
            SectionType.OPENING_REMARKS: [SectionType.QA_SESSION, SectionType.CLOSING_REMARKS],
            SectionType.QA_SESSION: [SectionType.CLOSING_REMARKS, SectionType.QA_SESSION],
            SectionType.CLOSING_REMARKS: [],  # Terminal state
            SectionType.TRANSITION: [SectionType.OPENING_REMARKS, SectionType.QA_SESSION, SectionType.CLOSING_REMARKS],
            SectionType.UNKNOWN: [SectionType.OPENING_REMARKS, SectionType.QA_SESSION, SectionType.CLOSING_REMARKS],
        }
        return new_state in valid_transitions.get(self.current_state, [])

    def transition(self, new_state: SectionType, offset: int, reason: str):
        """Execute state transition."""
        self.transitions.append({
            "from": self.current_state.value,
            "to": new_state.value,
            "offset": offset,
            "reason": reason,
        })
        self.current_state = new_state


# =============================================================================
# Helper Functions
# =============================================================================

def _is_valid_speaker_name(name: str) -> bool:
    """Validate speaker name (deterministic check)."""
    if not name or '\n' in name:
        return False

    name = name.strip()
    if len(name) < 2 or len(name) > 50:
        return False

    words = name.split()
    if len(words) > 4:
        return False

    first_word = words[0].lower().rstrip('.,;:')
    if first_word in CONTENT_INDICATOR_WORDS:
        return False

    content_word_count = sum(1 for w in words if w.lower().rstrip('.,;:') in CONTENT_INDICATOR_WORDS)
    if content_word_count >= 2:
        return False

    if name.rstrip().endswith(('.', '!', '?', ',')):
        return False

    return True


def _find_pattern_candidates(
    text: str,
    patterns: list[tuple[str, str]],
) -> list[BoundaryCandidate]:
    """Find all pattern matches as candidates."""
    candidates = []

    for pattern, name in patterns:
        for match in re.finditer(pattern, text):
            # Get snippet around match - increased for better context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 150)
            snippet = text[start:end].replace('\n', ' ')

            candidates.append(BoundaryCandidate(
                char_offset=match.start(),
                text_snippet=snippet,
                signal_type=DetectionSignal.SECTION_HEADER,
                pattern_matched=name,
            ))

    return sorted(candidates, key=lambda c: c.char_offset)


def _detect_speakers(text: str) -> list[tuple[int, str]]:
    """Detect valid speaker labels in text."""
    speakers = []
    for match in SPEAKER_LABEL_PATTERN.finditer(text):
        speaker = match.group("speaker").strip()
        if _is_valid_speaker_name(speaker):
            speakers.append((match.start(), speaker))
    return speakers


@dataclass
class QABlock:
    """A regex-detected Q&A block (investor introduction to next introduction)."""
    block_id: str
    start_offset: int
    end_offset: int
    questioner_name: str
    questioner_company: Optional[str]
    raw_text: str
    start_page: int
    end_page: int
    introduction_pattern: str


def detect_qa_blocks(
    qa_section_text: str,
    section_start_offset: int,
    section_start_page: int,
    section_end_page: int,
    text_length: int,
    total_pages: int,
) -> list[QABlock]:
    """Detect Q&A blocks within a Q&A section.

    Each block starts when moderator introduces a new questioner.
    Ends when next questioner is introduced or section ends.

    REGEX PROPOSES - never misses a block.
    LLM will structure each block internally.

    Args:
        qa_section_text: Text of the Q&A section
        section_start_offset: Character offset where section starts
        section_start_page: Page where section starts
        section_end_page: Page where section ends
        text_length: Total document length
        total_pages: Total pages in document

    Returns:
        List of QABlock objects
    """
    blocks = []
    block_starts = []

    # Find all investor introductions
    for pattern, pattern_name in QA_BLOCK_START_PATTERNS:
        for match in re.finditer(pattern, qa_section_text):
            questioner_name = match.group(1).strip() if match.lastindex >= 1 else "Unknown"
            questioner_company = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else None

            # Clean up questioner name
            questioner_name = re.sub(r'\s+', ' ', questioner_name).strip()
            if questioner_company:
                questioner_company = re.sub(r'\s+', ' ', questioner_company).strip()

            block_starts.append({
                "offset": match.start(),
                "questioner_name": questioner_name,
                "questioner_company": questioner_company,
                "pattern": pattern_name,
                "match_text": match.group(0),
            })

    # Sort by offset and deduplicate
    block_starts = sorted(block_starts, key=lambda x: x["offset"])

    # Remove duplicates (same questioner at similar offset)
    deduped_starts = []
    for start in block_starts:
        if not deduped_starts:
            deduped_starts.append(start)
        elif start["offset"] - deduped_starts[-1]["offset"] > 50:
            deduped_starts.append(start)
        elif start["questioner_name"].lower() != deduped_starts[-1]["questioner_name"].lower():
            deduped_starts.append(start)

    logger.debug("qa_block_starts_detected", count=len(deduped_starts))

    # Create blocks from starts
    for i, start in enumerate(deduped_starts):
        # End is either next block start or section end
        if i + 1 < len(deduped_starts):
            end_offset = deduped_starts[i + 1]["offset"]
        else:
            end_offset = len(qa_section_text)

        block_text = qa_section_text[start["offset"]:end_offset].strip()

        if len(block_text) < 50:
            continue

        # Estimate pages
        abs_start = section_start_offset + start["offset"]
        abs_end = section_start_offset + end_offset
        start_page = _estimate_page(abs_start, text_length, total_pages)
        end_page = _estimate_page(abs_end, text_length, total_pages)

        blocks.append(QABlock(
            block_id=f"qa_block_{len(blocks):03d}",
            start_offset=start["offset"],
            end_offset=end_offset,
            questioner_name=start["questioner_name"],
            questioner_company=start["questioner_company"],
            raw_text=block_text,
            start_page=max(start_page, section_start_page),
            end_page=min(end_page, section_end_page),
            introduction_pattern=start["pattern"],
        ))

    logger.info("qa_blocks_detected", count=len(blocks))
    return blocks


def _estimate_page(offset: int, text_length: int, total_pages: int) -> int:
    """Estimate page number from character offset."""
    if total_pages <= 1 or text_length == 0:
        return 1
    chars_per_page = text_length / total_pages
    return min(int(offset / chars_per_page) + 1, total_pages)


# =============================================================================
# Main Boundary Detection (Hybrid Approach)
# =============================================================================

def detect_boundaries(
    full_text: str,
    total_pages: int,
    use_llm_confirmation: bool = True,
    page_offsets: Optional[list[int]] = None,
) -> tuple[BoundaryDetectionResult, BoundaryDetectionTrace]:
    """Detect section boundaries using hybrid intelligence.

    Approach:
    1. Deterministic pattern matching for HIGH-RECALL candidates
    2. LLM confirmation for each candidate (if enabled)
    3. State machine enforces valid transitions
    4. Returns inspectable trace

    Args:
        full_text: Complete transcript text.
        total_pages: Total number of pages.
        use_llm_confirmation: Whether to use LLM for confirmation.
        page_offsets: Optional page offset mapping.

    Returns:
        Tuple of (BoundaryDetectionResult, BoundaryDetectionTrace)
    """
    logger.info("boundary_detection_start", text_length=len(full_text), use_llm=use_llm_confirmation)

    text_length = len(full_text)
    trace = BoundaryDetectionTrace()
    state_machine = SectionStateMachine()

    # Step 1: Find EXPLICIT Q&A start markers first (highest priority)
    explicit_qa_candidates = _find_pattern_candidates(full_text, EXPLICIT_QA_START_PATTERNS)
    trace.candidates.extend(explicit_qa_candidates)
    logger.debug("explicit_qa_candidates_found", count=len(explicit_qa_candidates))

    # Step 2: Find loose Q&A start candidates (fallback)
    qa_candidates = _find_pattern_candidates(full_text, QA_START_CANDIDATES)
    trace.candidates.extend(qa_candidates)
    logger.debug("qa_candidates_found", count=len(qa_candidates))

    # Combine: explicit patterns first, then fallback
    all_qa_candidates = explicit_qa_candidates + [c for c in qa_candidates if c.char_offset not in [e.char_offset for e in explicit_qa_candidates]]

    # Step 2: Find closing candidates
    closing_candidates = _find_pattern_candidates(full_text, CLOSING_CANDIDATES)
    trace.candidates.extend(closing_candidates)
    logger.debug("closing_candidates_found", count=len(closing_candidates))

    # Step 3: LLM confirmation for Q&A start (use first strong candidate)
    # Explicit patterns are trusted without LLM confirmation
    qa_start_offset = None

    # First check explicit patterns (no LLM needed)
    for candidate in explicit_qa_candidates:
        if candidate.char_offset < text_length * 0.05:  # Skip if too early
            continue
        qa_start_offset = candidate.char_offset
        candidate.llm_confirmed = True
        candidate.llm_confidence = 0.95
        candidate.llm_reasoning = "Explicit Q&A start marker detected"
        state_machine.transition(
            SectionType.QA_SESSION,
            qa_start_offset,
            f"Explicit pattern: {candidate.pattern_matched}"
        )
        logger.info("explicit_qa_start_found", offset=qa_start_offset, pattern=candidate.pattern_matched)
        break

    # If no explicit match, fall back to loose patterns with LLM
    if qa_start_offset is None:
        for candidate in all_qa_candidates:
            # Skip if too early (likely part of title/header)
            if candidate.char_offset < text_length * 0.1:
                continue

            if use_llm_confirmation:
                context_before = full_text[max(0, candidate.char_offset - 200):candidate.char_offset]
                decision = confirm_qa_session_start(candidate.text_snippet, context_before)
                candidate.llm_confirmed = decision.decision == "YES"
                candidate.llm_confidence = decision.confidence
                candidate.llm_reasoning = decision.reasoning

                if candidate.llm_confirmed and decision.confidence >= 0.6:
                    qa_start_offset = candidate.char_offset
                    state_machine.transition(
                        SectionType.QA_SESSION,
                        qa_start_offset,
                        f"LLM confirmed: {candidate.pattern_matched}"
                    )
                    break
            else:
                # Without LLM, use first candidate after 10% of document
                qa_start_offset = candidate.char_offset
                state_machine.transition(
                    SectionType.QA_SESSION,
                    qa_start_offset,
                    f"Pattern match: {candidate.pattern_matched}"
                )
                break

    # Fallback: if no Q&A start found, estimate at 30%
    if qa_start_offset is None:
        qa_start_offset = int(text_length * 0.3)
        state_machine.transition(
            SectionType.QA_SESSION,
            qa_start_offset,
            "Fallback: 30% of document"
        )
        logger.warning("qa_start_not_found_using_fallback")

    # Step 4: Find closing section
    closing_offset = text_length
    for candidate in closing_candidates:
        # Must be in last 25% of document
        if candidate.char_offset < text_length * 0.75:
            continue

        if use_llm_confirmation:
            decision = confirm_section_boundary(candidate.text_snippet, "qa_session")
            candidate.llm_confirmed = decision.decision == "YES"
            candidate.llm_confidence = decision.confidence
            candidate.llm_reasoning = decision.reasoning

            if candidate.llm_confirmed:
                closing_offset = candidate.char_offset
                state_machine.transition(
                    SectionType.CLOSING_REMARKS,
                    closing_offset,
                    f"LLM confirmed: {candidate.pattern_matched}"
                )
                break
        else:
            closing_offset = candidate.char_offset
            state_machine.transition(
                SectionType.CLOSING_REMARKS,
                closing_offset,
                f"Pattern match: {candidate.pattern_matched}"
            )
            break

    trace.state_transitions = state_machine.transitions

    # Step 5: Build sections from boundaries
    sections = []
    boundaries = [0, qa_start_offset, closing_offset, text_length]
    boundaries = sorted(set(b for b in boundaries if 0 <= b <= text_length))
    trace.final_boundaries = boundaries

    section_types = [SectionType.OPENING_REMARKS, SectionType.QA_SESSION, SectionType.CLOSING_REMARKS]
    type_index = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        section_text = full_text[start:end].strip()

        if not section_text or len(section_text) < 50:
            continue

        # Determine section type
        if start < qa_start_offset:
            section_type = SectionType.OPENING_REMARKS
        elif start >= closing_offset:
            section_type = SectionType.CLOSING_REMARKS
        else:
            section_type = SectionType.QA_SESSION

        # Detect speakers in section
        speakers = _detect_speakers(section_text)
        speaker_names = list(dict.fromkeys([s[1] for s in speakers]))

        sections.append(TranscriptSection(
            section_id=f"section_{len(sections):03d}",
            section_type=section_type,
            start_page=_estimate_page(start, text_length, total_pages),
            end_page=_estimate_page(end, text_length, total_pages),
            char_offset_start=start,
            char_offset_end=end,
            raw_text=section_text,
            detected_speakers=speaker_names,
            detection_signals=[DetectionSignal.SECTION_HEADER],
            detection_confidence=0.8,
            sequence_number=len(sections),
        ))

    # Step 6: Segment Q&A section by moderator cues
    qa_sections = [s for s in sections if s.section_type == SectionType.QA_SESSION]
    if qa_sections and len(qa_sections) == 1:
        qa_section = qa_sections[0]
        moderator_cues = _find_pattern_candidates(qa_section.raw_text, MODERATOR_CUE_PATTERNS)

        if len(moderator_cues) > 1:
            # Sub-segment the Q&A section
            new_sections = []
            prev_offset = 0

            for cue in moderator_cues[1:]:  # Skip first
                if cue.char_offset - prev_offset < 100:
                    continue

                sub_text = qa_section.raw_text[prev_offset:cue.char_offset].strip()
                if sub_text and len(sub_text) > 50:
                    speakers = _detect_speakers(sub_text)
                    new_sections.append(TranscriptSection(
                        section_id=f"section_{len(sections) + len(new_sections):03d}",
                        section_type=SectionType.QA_SESSION,
                        start_page=qa_section.start_page,
                        end_page=qa_section.end_page,
                        char_offset_start=qa_section.char_offset_start + prev_offset,
                        char_offset_end=qa_section.char_offset_start + cue.char_offset,
                        raw_text=sub_text,
                        detected_speakers=list(dict.fromkeys([s[1] for s in speakers])),
                        detection_signals=[DetectionSignal.MODERATOR_CUE],
                        detection_confidence=0.75,
                        sequence_number=0,
                    ))
                prev_offset = cue.char_offset

            # Add final segment
            if prev_offset < len(qa_section.raw_text):
                final_text = qa_section.raw_text[prev_offset:].strip()
                if final_text and len(final_text) > 50:
                    speakers = _detect_speakers(final_text)
                    new_sections.append(TranscriptSection(
                        section_id=f"section_{len(sections) + len(new_sections):03d}",
                        section_type=SectionType.QA_SESSION,
                        start_page=qa_section.start_page,
                        end_page=qa_section.end_page,
                        char_offset_start=qa_section.char_offset_start + prev_offset,
                        char_offset_end=qa_section.char_offset_end,
                        raw_text=final_text,
                        detected_speakers=list(dict.fromkeys([s[1] for s in speakers])),
                        detection_signals=[DetectionSignal.MODERATOR_CUE],
                        detection_confidence=0.75,
                        sequence_number=0,
                    ))

            if new_sections:
                # Replace single Q&A section with sub-segments
                sections = [s for s in sections if s.section_type != SectionType.QA_SESSION]
                sections.extend(new_sections)

    # Renumber sections
    sections = sorted(sections, key=lambda s: s.char_offset_start)
    for i, section in enumerate(sections):
        section.section_id = f"section_{i:03d}"
        section.sequence_number = i

    # Compute stats
    qa_count = sum(1 for s in sections if s.section_type == SectionType.QA_SESSION)
    opening_count = sum(1 for s in sections if s.section_type == SectionType.OPENING_REMARKS)
    covered = sum(s.char_offset_end - s.char_offset_start for s in sections)
    coverage = (covered / text_length * 100) if text_length > 0 else 0

    result = BoundaryDetectionResult(
        sections=sections,
        total_sections=len(sections),
        qa_section_count=qa_count,
        opening_remarks_count=opening_count,
        coverage_percent=coverage,
        gaps_detected=0,
    )

    logger.info(
        "boundary_detection_complete",
        total_sections=len(sections),
        qa_sections=qa_count,
        coverage=round(coverage, 1),
        llm_confirmations=sum(1 for c in trace.candidates if c.llm_confirmed is not None),
    )

    return result, trace
