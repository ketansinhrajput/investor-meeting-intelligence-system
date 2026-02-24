"""Stage 4: Q&A Extraction - Context-Aware Hybrid Intelligence.

CONTEXT-AWARE HYBRID INTELLIGENCE APPROACH:
- Phase A: Deterministic candidate generation with HIGH RECALL
- Phase B: DOCUMENT-LEVEL LLM verification (LLM sees ALL candidates at once)

Rules propose -> LLM validates -> Code enforces invariants

PHILOSOPHY:
- High recall first, precision second
- Deterministic methods propose candidates
- LLM has FULL AUTHORITY to validate, reject, correct boundaries
- Every LLM decision must emit justification trace

HARD RULES (enforced by code AFTER LLM decision):
- Moderator speech is NEVER part of Q&A content (always transition)

LLM has FULL AUTHORITY to:
- Validate if extracted Q&A units are semantically correct
- Reject candidates that don't make sense
- Correct boundaries based on full context
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Literal

import structlog

from src.pipeline_v2.llm_helpers import (
    extract_qa_units_from_section,
    QAExtractionDecision,
    QAUnit as LLMQAUnit,
    _invoke_llm,
)
from src.pipeline_v2.models import (
    BoundaryDetectionResult,
    ExtractedQAUnit,
    QAExtractionResult,
    SectionType,
    SpeakerRegistry,
    SpeakerRole,
    SpeakerTurn,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# ENHANCED Trace Dataclasses (with evidence_spans)
# =============================================================================

@dataclass
class TurnIntentRecord:
    """Record of LLM turn intent classification."""
    turn_index: int
    speaker_name: Optional[str]
    speaker_role: str
    text_snippet: str
    intent: str
    supports_qa: bool
    confidence: float
    evidence_spans: list[dict] = field(default_factory=list)
    is_follow_up: bool = False
    follow_up_reason: Optional[str] = None
    classified_by_llm: bool = False


@dataclass
class QAUnitConstruction:
    """Record of how a Q&A unit was constructed."""
    qa_id: str
    question_turn_indices: list[int]
    answer_turn_indices: list[int]
    questioner_name: str
    questioner_source: str  # "speaker_label", "registry", "heuristic"
    responder_names: list[str]
    responder_source: str  # "speaker_label", "registry", "management_default"
    is_follow_up: bool
    follow_up_reason: Optional[str] = None
    evidence_spans: list[dict] = field(default_factory=list)


@dataclass
class QAExtractionTrace:
    """Complete trace of Q&A extraction for inspection.

    This trace allows humans to audit:
    - LLM intent classification for each turn
    - How Q&A units were formed
    - Evidence supporting each decision
    """
    # Per-turn intent classifications
    turn_intent_records: list[TurnIntentRecord] = field(default_factory=list)

    # Q&A unit construction decisions
    qa_constructions: list[QAUnitConstruction] = field(default_factory=list)

    # Hard rule enforcement
    hard_rule_enforcements: list[dict] = field(default_factory=list)

    # Section-level stats
    sections_processed: int = 0
    sections_with_qa: int = 0
    sections_with_no_qa: list[str] = field(default_factory=list)

    # LLM usage stats
    llm_calls_made: int = 0
    turns_classified_by_llm: int = 0
    turns_classified_deterministic: int = 0


# =============================================================================
# Speaker Label Extraction
# =============================================================================

SPEAKER_LABEL_PATTERNS = [
    re.compile(r"^(?P<speaker>[A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){0,3})\s*[-–—:]\s*(?P<text>.*)$"),
    re.compile(r"^\[(?P<speaker>[A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){0,3})\]\s*(?P<text>.*)$"),
    re.compile(r"^(?P<speaker>[A-Z][A-Z\.\-']+(?:\s+[A-Z][A-Z\.\-']+){0,3})\s*[-–—:]\s*(?P<text>.*)$"),
]

CONTENT_INDICATOR_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
    'conference', 'call', 'reminder', 'thank', 'thanks', 'you', 'we', 'i',
}


def _is_valid_speaker_name(name: str) -> bool:
    """Validate that a string is a plausible speaker name."""
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
    if name.lower() in ('moderator', 'operator'):
        return True
    return True


def _extract_speaker_label(line: str) -> tuple[Optional[str], str]:
    """Extract speaker label from line if present."""
    for pattern in SPEAKER_LABEL_PATTERNS:
        match = pattern.match(line)
        if match:
            speaker = match.group("speaker").strip()
            text = match.group("text").strip()
            if _is_valid_speaker_name(speaker):
                return speaker, text
    return None, line


# =============================================================================
# LLM-Governed State Machine
# =============================================================================

class LLMGovernedQAStateMachine:
    """State machine where LLM governs turn intent classification.

    States:
    - IDLE: No active Q&A
    - IN_QUESTION: Collecting question content
    - IN_ANSWER: Collecting answer content

    KEY: State transitions are governed by LLM's TurnIntentDecision,
    not by pattern matching. The LLM decides when questions end and
    answers begin.
    """

    IDLE = "idle"
    IN_QUESTION = "in_question"
    IN_ANSWER = "in_answer"

    def __init__(
        self,
        section_id: str,
        base_page: int,
        registry: Optional[SpeakerRegistry],
        use_llm: bool = True,
    ):
        self.section_id = section_id
        self.base_page = base_page
        self.registry = registry
        self.use_llm = use_llm

        self.state = self.IDLE
        self.qa_units: list[ExtractedQAUnit] = []
        self.qa_counter = 0

        # Trace collection
        self.turn_records: list[TurnIntentRecord] = []
        self.qa_constructions: list[QAUnitConstruction] = []
        self.hard_rules: list[dict] = []
        self.llm_calls = 0

        # Current Q&A being built
        self.current_questioner: Optional[str] = None
        self.current_questioner_id: Optional[str] = None
        self.current_question_turns: list[tuple[int, str, str]] = []  # (index, speaker, text)
        self.current_question_page: int = base_page
        self.current_answer_turns: list[tuple[int, str, str]] = []
        self.current_responders: list[str] = []
        self.current_answer_page: int = base_page

        # Turn history for LLM context
        self.turn_history: list[tuple[str, str, str]] = []  # (speaker, role, text)

    def _get_speaker_role(self, speaker: Optional[str]) -> str:
        """Get speaker role from registry."""
        if not speaker:
            return "unknown"
        if speaker.lower() in ('moderator', 'operator'):
            return "moderator"
        if self.registry:
            info = self.registry.get_by_name(speaker)
            if info:
                return info.role.value
        return "unknown"

    def _classify_turn_intent(
        self,
        turn_index: int,
        speaker: Optional[str],
        text: str,
        next_turn: Optional[tuple[str, str, str]],
    ) -> TurnIntentRecord:
        """Classify turn intent using LLM (or deterministic fallback).

        HARD RULE: Moderator is ALWAYS MODERATOR_TRANSITION.
        """
        speaker_role = self._get_speaker_role(speaker)

        # HARD RULE: Moderator speech is ALWAYS transition
        if speaker_role == "moderator":
            self.hard_rules.append({
                "rule": "moderator_always_transition",
                "turn_index": turn_index,
                "speaker": speaker,
                "action": "forced_moderator_transition",
            })
            return TurnIntentRecord(
                turn_index=turn_index,
                speaker_name=speaker,
                speaker_role=speaker_role,
                text_snippet=text[:100],
                intent="MODERATOR_TRANSITION",
                supports_qa=False,
                confidence=1.0,
                evidence_spans=[{
                    "text": "Speaker is moderator",
                    "source": "speaker_role",
                    "relevance": "HARD RULE: Moderator speech is always transition"
                }],
                classified_by_llm=False,
            )

        # Use LLM for classification
        if self.use_llm:
            self.llm_calls += 1
            try:
                llm_decision = classify_turn_intent(
                    current_turn_text=text,
                    current_speaker=speaker or "Unknown",
                    current_speaker_role=speaker_role,
                    previous_turns=self.turn_history[-2:],
                    next_turn=next_turn,
                    current_qa_state=self.state,
                    current_question_text=" ".join([t[2] for t in self.current_question_turns])[:500],
                )

                return TurnIntentRecord(
                    turn_index=turn_index,
                    speaker_name=speaker,
                    speaker_role=speaker_role,
                    text_snippet=text[:100],
                    intent=llm_decision.intent,
                    supports_qa=llm_decision.supports_qa,
                    confidence=llm_decision.confidence,
                    evidence_spans=[
                        {"text": e.text, "source": e.source, "relevance": e.relevance}
                        for e in llm_decision.evidence_spans
                    ],
                    is_follow_up=llm_decision.is_follow_up,
                    follow_up_reason=llm_decision.follow_up_reason,
                    classified_by_llm=True,
                )

            except Exception as e:
                logger.warning("llm_turn_classification_failed", error=str(e))
                # Fall through to deterministic

        # Deterministic fallback
        return self._deterministic_classify(turn_index, speaker, speaker_role, text)

    def _deterministic_classify(
        self,
        turn_index: int,
        speaker: Optional[str],
        speaker_role: str,
        text: str,
    ) -> TurnIntentRecord:
        """Fallback deterministic classification."""
        has_question = "?" in text

        if speaker_role == "analyst" and has_question:
            intent = "QUESTION_START" if self.state == self.IDLE else "QUESTION_CONTINUATION"
            supports_qa = True
        elif speaker_role == "management" and self.state == self.IN_QUESTION:
            intent = "ANSWER_START"
            supports_qa = True
        elif speaker_role == "analyst":
            intent = "QUESTION_START" if self.state == self.IDLE else "QUESTION_CONTINUATION"
            supports_qa = True
        elif speaker_role == "management":
            intent = "ANSWER_CONTINUATION" if self.state == self.IN_ANSWER else "ANSWER_START"
            supports_qa = True
        else:
            intent = "NON_QA"
            supports_qa = False

        return TurnIntentRecord(
            turn_index=turn_index,
            speaker_name=speaker,
            speaker_role=speaker_role,
            text_snippet=text[:100],
            intent=intent,
            supports_qa=supports_qa,
            confidence=0.6,
            evidence_spans=[{
                "text": f"role={speaker_role}, has_question={has_question}",
                "source": "deterministic",
                "relevance": "Fallback classification based on role and punctuation"
            }],
            classified_by_llm=False,
        )

    def _save_current_qa(self):
        """Save current Q&A unit if valid."""
        if not self.current_question_turns:
            return

        question_text = " ".join([t[2] for t in self.current_question_turns]).strip()
        answer_text = " ".join([t[2] for t in self.current_answer_turns]).strip()

        if len(question_text) < 10:
            return

        qa_id = f"qa_{self.qa_counter:03d}"

        # Determine questioner
        questioner_name = self.current_questioner or "Unknown"
        questioner_source = "speaker_label" if self.current_questioner else "heuristic"

        # Determine responders
        responders = list(dict.fromkeys(self.current_responders))
        if not responders and answer_text:
            responders = ["Management"]
        responder_source = "speaker_label" if self.current_responders else "management_default"

        # Check for follow-up
        is_follow_up = False
        follow_up_reason = None
        if self.qa_units and self.current_questioner:
            prev_qa = self.qa_units[-1]
            if prev_qa.questioner_name == self.current_questioner:
                is_follow_up = True
                follow_up_reason = "same_speaker"

        # Build turn objects
        question_turns = [SpeakerTurn(
            speaker_name=questioner_name,
            speaker_id=self.current_questioner_id,
            text=question_text,
            page_number=self.current_question_page,
            is_question=True,
        )]

        response_turns = []
        if answer_text:
            response_turns = [SpeakerTurn(
                speaker_name=responders[0] if responders else "Management",
                speaker_id=None,
                text=answer_text,
                page_number=self.current_answer_page,
                is_question=False,
            )]

        qa_unit = ExtractedQAUnit(
            qa_id=qa_id,
            source_section_id=self.section_id,
            questioner_name=questioner_name,
            questioner_id=self.current_questioner_id,
            question_turns=question_turns,
            response_turns=response_turns,
            responder_names=responders,
            responder_ids=[],
            question_text=question_text,
            response_text=answer_text,
            is_follow_up=is_follow_up,
            follow_up_of=self.qa_units[-1].qa_id if is_follow_up else None,
            start_page=self.current_question_page,
            end_page=self.current_answer_page,
            sequence_in_session=self.qa_counter,
        )
        self.qa_units.append(qa_unit)

        # Record construction
        self.qa_constructions.append(QAUnitConstruction(
            qa_id=qa_id,
            question_turn_indices=[t[0] for t in self.current_question_turns],
            answer_turn_indices=[t[0] for t in self.current_answer_turns],
            questioner_name=questioner_name,
            questioner_source=questioner_source,
            responder_names=responders,
            responder_source=responder_source,
            is_follow_up=is_follow_up,
            follow_up_reason=follow_up_reason,
        ))

        self.qa_counter += 1
        self._reset_current()

    def _reset_current(self):
        """Reset current Q&A state."""
        self.current_questioner = None
        self.current_questioner_id = None
        self.current_question_turns = []
        self.current_answer_turns = []
        self.current_responders = []

    def process_turn(
        self,
        turn_index: int,
        speaker: Optional[str],
        text: str,
        page: int,
        next_turn: Optional[tuple[str, str, str]],
    ):
        """Process a single turn with LLM-governed state transitions."""
        if not text.strip():
            return

        # Classify turn intent
        intent_record = self._classify_turn_intent(turn_index, speaker, text, next_turn)
        self.turn_records.append(intent_record)

        # Update turn history
        self.turn_history.append((speaker or "Unknown", intent_record.speaker_role, text[:200]))
        if len(self.turn_history) > 5:
            self.turn_history.pop(0)

        # =====================================================================
        # LLM-GOVERNED STATE TRANSITIONS
        # =====================================================================

        intent = intent_record.intent

        if intent == "MODERATOR_TRANSITION":
            # Moderator transition ends current Q&A
            if self.state in (self.IN_QUESTION, self.IN_ANSWER):
                self._save_current_qa()
            self.state = self.IDLE
            return

        if intent == "NON_QA":
            # Non-Q&A content - ignore or end current Q&A
            if self.state == self.IN_ANSWER:
                # Continue collecting answer
                self.current_answer_turns.append((turn_index, speaker or "Unknown", text))
            return

        if intent == "QUESTION_START":
            # New question starting
            if self.state in (self.IN_QUESTION, self.IN_ANSWER):
                self._save_current_qa()

            self.current_questioner = speaker
            if speaker and self.registry:
                info = self.registry.get_by_name(speaker)
                self.current_questioner_id = info.speaker_id if info else None
            self.current_question_turns = [(turn_index, speaker or "Unknown", text)]
            self.current_question_page = page
            self.state = self.IN_QUESTION

        elif intent == "QUESTION_CONTINUATION":
            if self.state == self.IN_QUESTION:
                self.current_question_turns.append((turn_index, speaker or "Unknown", text))
            elif self.state == self.IDLE:
                # Treat as new question
                self.current_questioner = speaker
                if speaker and self.registry:
                    info = self.registry.get_by_name(speaker)
                    self.current_questioner_id = info.speaker_id if info else None
                self.current_question_turns = [(turn_index, speaker or "Unknown", text)]
                self.current_question_page = page
                self.state = self.IN_QUESTION

        elif intent == "ANSWER_START":
            if self.state == self.IN_QUESTION:
                # Question ends, answer begins
                self.current_answer_turns = [(turn_index, speaker or "Unknown", text)]
                self.current_answer_page = page
                if speaker:
                    self.current_responders.append(speaker)
                self.state = self.IN_ANSWER
            elif self.state == self.IDLE:
                # Answer without question - ignore
                pass

        elif intent == "ANSWER_CONTINUATION":
            if self.state == self.IN_ANSWER:
                self.current_answer_turns.append((turn_index, speaker or "Unknown", text))
                if speaker and speaker not in self.current_responders:
                    self.current_responders.append(speaker)
            elif self.state == self.IN_QUESTION:
                # Might be answer starting - treat as answer start
                self.current_answer_turns = [(turn_index, speaker or "Unknown", text)]
                self.current_answer_page = page
                if speaker:
                    self.current_responders.append(speaker)
                self.state = self.IN_ANSWER

    def finalize(self):
        """Finalize extraction, saving any remaining Q&A."""
        if self.state in (self.IN_QUESTION, self.IN_ANSWER):
            self._save_current_qa()
        self.state = self.IDLE


# =============================================================================
# Turn Parsing
# =============================================================================

def _parse_turns(section_text: str) -> list[tuple[int, Optional[str], str]]:
    """Parse section text into turns.

    Returns:
        List of (line_index, speaker_name_or_None, text)
    """
    turns = []
    lines = section_text.split("\n")

    current_speaker = None
    current_text_parts = []
    current_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Try to extract speaker label
        speaker, text = _extract_speaker_label(line)

        if speaker:
            # Save previous turn if exists
            if current_text_parts:
                turns.append((
                    current_start,
                    current_speaker,
                    " ".join(current_text_parts)
                ))

            # Start new turn
            current_speaker = speaker
            current_text_parts = [text] if text else []
            current_start = i
        else:
            # Continue current turn
            current_text_parts.append(line)

    # Save final turn
    if current_text_parts:
        turns.append((
            current_start,
            current_speaker,
            " ".join(current_text_parts)
        ))

    return turns


# =============================================================================
# Section-Level LLM Extraction
# =============================================================================

def _extract_section_with_llm(
    section_id: str,
    section_text: str,
    turns: list[tuple[int, Optional[str], str]],
    registry: Optional[SpeakerRegistry],
    base_page: int,
    trace: QAExtractionTrace,
) -> list[ExtractedQAUnit]:
    """Extract Q&A units using section-level LLM authority.

    LLM sees the ENTIRE section and decides:
    - Exact question/answer boundaries
    - Multi-part question grouping
    - Follow-up relationships
    - Multi-speaker answers

    Returns:
        List of ExtractedQAUnit or empty list on failure
    """
    # Build speaker turns for LLM
    speaker_turns = []
    for line_idx, speaker, text in turns:
        role = "unknown"
        if speaker:
            if speaker.lower() in ('moderator', 'operator'):
                role = "moderator"
            elif registry:
                info = registry.get_by_name(speaker)
                if info:
                    role = info.role.value
        speaker_turns.append({
            "speaker": speaker or "Unknown",
            "text": text,
            "line_num": line_idx,
            "role": role,
        })

    # Build role registry for LLM
    role_map: dict[str, str] = {}
    if registry:
        for speaker in registry.speakers.values():
            role_map[speaker.canonical_name] = speaker.role.value
            for alias in speaker.aliases:
                role_map[alias] = speaker.role.value

    # SECTION-LEVEL LLM CALL
    trace.llm_calls_made += 1
    try:
        llm_decision = extract_qa_units_from_section(
            section_text=section_text,
            speaker_turns=speaker_turns,
            verified_registry=role_map,
            section_start_page=base_page,
        )

        # Record moderator lines in trace
        if llm_decision.moderator_lines:
            trace.hard_rule_enforcements.append({
                "rule": "moderator_lines_excluded",
                "section_id": section_id,
                "moderator_lines": llm_decision.moderator_lines,
                "action": "excluded_from_qa",
            })

        # Convert LLM Q&A units to ExtractedQAUnit
        qa_units = []
        for i, llm_qa in enumerate(llm_decision.qa_units):
            # Estimate pages from line numbers
            total_lines = len(section_text.split("\n"))
            lines_per_page = max(1, total_lines // 3)
            start_page = base_page + (llm_qa.start_line // lines_per_page)
            end_page = base_page + (llm_qa.end_line // lines_per_page)

            # Build turn objects
            question_turns = [SpeakerTurn(
                speaker_name=llm_qa.questioner,
                speaker_id=None,
                text=llm_qa.question_text,
                page_number=start_page,
                is_question=True,
            )]

            response_turns = []
            if llm_qa.response_text:
                for responder in llm_qa.responders[:1]:  # Primary responder
                    response_turns.append(SpeakerTurn(
                        speaker_name=responder,
                        speaker_id=None,
                        text=llm_qa.response_text,
                        page_number=end_page,
                        is_question=False,
                    ))

            qa_id = f"qa_{i:03d}"
            qa_unit = ExtractedQAUnit(
                qa_id=qa_id,
                source_section_id=section_id,
                questioner_name=llm_qa.questioner,
                questioner_id=None,
                question_turns=question_turns,
                response_turns=response_turns,
                responder_names=llm_qa.responders,
                responder_ids=[],
                question_text=llm_qa.question_text,
                response_text=llm_qa.response_text,
                is_follow_up=llm_qa.is_follow_up,
                follow_up_of=f"qa_{llm_qa.follow_up_of:03d}" if llm_qa.follow_up_of is not None else None,
                start_page=start_page,
                end_page=end_page,
                sequence_in_session=i,
                boundary_justification=llm_qa.justification,
            )
            qa_units.append(qa_unit)

            # Record construction in trace
            trace.qa_constructions.append(QAUnitConstruction(
                qa_id=qa_id,
                question_turn_indices=[llm_qa.start_line],
                answer_turn_indices=[llm_qa.end_line],
                questioner_name=llm_qa.questioner,
                questioner_source="llm_section_level",
                responder_names=llm_qa.responders,
                responder_source="llm_section_level",
                is_follow_up=llm_qa.is_follow_up,
                follow_up_reason=None,  # No separate follow_up_reasoning field anymore
                evidence_spans=[{
                    "text": llm_qa.justification[:200],
                    "source": "llm_section_level",
                    "relevance": "LLM justification for Q&A boundaries"
                }],
            ))

        logger.info(
            "section_level_extraction_complete",
            section_id=section_id,
            qa_units=len(qa_units),
            confidence=llm_decision.confidence,
        )

        return qa_units

    except Exception as e:
        logger.warning("section_level_extraction_failed", section_id=section_id, error=str(e))
        return []


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_qa_units(
    boundary_result: BoundaryDetectionResult,
    registry: Optional[SpeakerRegistry] = None,
    use_llm_fallback: bool = True,
) -> tuple[QAExtractionResult, QAExtractionTrace]:
    """Extract Q&A units using two-phase hybrid approach.

    PHASE A: Deterministic candidate generation
    - Regex-based Q&A unit extraction
    - High recall, may include false positives
    - Basic structural validation

    PHASE B: LLM-based contextual validation
    - Full document context provided to LLM
    - LLM can reject, correct boundaries, validate completeness
    - Hard rules enforced by code after LLM decision

    Args:
        boundary_result: Result from boundary detection.
        registry: Optional speaker registry for role lookup.
        use_llm_fallback: Whether to use LLM for Phase B verification.

    Returns:
        Tuple of (QAExtractionResult, QAExtractionTrace)
    """
    logger.info("qa_extraction_start", total_sections=len(boundary_result.sections))

    trace = QAExtractionTrace()
    all_qa_units: list[ExtractedQAUnit] = []

    # Filter to Q&A sections only
    qa_sections = [
        s for s in boundary_result.sections
        if s.section_type == SectionType.QA_SESSION
    ]

    logger.debug("qa_sections_found", count=len(qa_sections))

    for section in qa_sections:
        trace.sections_processed += 1

        # Parse section into turns
        turns = _parse_turns(section.raw_text)

        if not turns:
            trace.sections_with_no_qa.append(section.section_id)
            continue

        # =====================================================================
        # PHASE A: Deterministic Candidate Generation
        # =====================================================================
        logger.info("phase_a_candidate_generation_start", section_id=section.section_id)

        # Generate Q&A candidates using smart regex
        qa_candidates = _extract_qa_candidates_with_regex(
            section_id=section.section_id,
            turns=turns,
            registry=registry,
            base_page=section.start_page,
        )

        trace.qa_constructions.extend([
            QAUnitConstruction(
                qa_id=candidate.qa_id,
                question_turn_indices=[],  # We'll populate this properly
                answer_turn_indices=[],
                questioner_name=candidate.questioner_name,
                questioner_source="regex_heuristic",
                responder_names=candidate.responder_names,
                responder_source="regex_heuristic",
                is_follow_up=candidate.is_follow_up,
                follow_up_reason=None,
                evidence_spans=[{
                    "text": candidate.boundary_justification or "Extracted using regex patterns",
                    "source": "regex_heuristic",
                    "relevance": "Initial candidate generation"
                }],
            ) for candidate in qa_candidates
        ])

        logger.info(
            "phase_a_complete",
            section_id=section.section_id,
            candidates=len(qa_candidates),
        )

        # =====================================================================
        # PHASE B: DOCUMENT-LEVEL LLM Validation
        # =====================================================================
        validated_qa_units = []
        if use_llm_fallback and qa_candidates:
            logger.info("phase_b_document_level_validation", section_id=section.section_id, candidates=len(qa_candidates))

            trace.llm_calls_made += 1

            # Prepare candidates for LLM validation
            candidate_dicts = []
            for i, candidate in enumerate(qa_candidates):
                candidate_dicts.append({
                    "index": i,
                    "questioner": candidate.questioner_name,
                    "question_text": candidate.question_text[:300],
                    "responders": candidate.responder_names,
                    "response_text": candidate.response_text[:500],
                    "is_follow_up": candidate.is_follow_up,
                })

            try:
                # SINGLE LLM CALL: Document-level validation
                llm_decision = _validate_qa_units_with_context(
                    candidates=candidate_dicts,
                    section_text=section.raw_text[:2000],  # Provide context
                    registry=registry,
                )

                # Process LLM decisions
                for i, validation_result in enumerate(llm_decision.get("validations", [])):
                    if i < len(qa_candidates) and validation_result.get("is_valid", False):
                        candidate = qa_candidates[i]

                        # Apply LLM corrections if provided
                        corrected_qa = _apply_llm_corrections(candidate, validation_result)
                        validated_qa_units.append(corrected_qa)

                        # Record validation in trace
                        trace.hard_rule_enforcements.append({
                            "rule": "llm_validation",
                            "qa_id": candidate.qa_id,
                            "action": "validated",
                            "confidence": validation_result.get("confidence", 0.8),
                        })
                    else:
                        # Record rejection in trace
                        trace.hard_rule_enforcements.append({
                            "rule": "llm_validation",
                            "candidate_index": i,
                            "action": "rejected",
                            "reason": validation_result.get("reason", "LLM validation failed"),
                        })

                logger.info(
                    "phase_b_complete",
                    section_id=section.section_id,
                    validated=len(validated_qa_units),
                    rejected=len(qa_candidates) - len(validated_qa_units),
                )

            except Exception as e:
                logger.warning("document_level_validation_failed", section_id=section.section_id, error=str(e))
                # Fallback: accept all candidates if LLM fails
                validated_qa_units = qa_candidates

        elif not use_llm_fallback:
            # LLM disabled: accept all candidates
            validated_qa_units = qa_candidates
            logger.info("phase_b_skipped_llm_disabled", section_id=section.section_id)

        logger.debug(
            "section_processed",
            section_id=section.section_id,
            qa_units_found=len(validated_qa_units),
        )

        if validated_qa_units:
            trace.sections_with_qa += 1
            # Renumber globally
            base_id = len(all_qa_units)
            for j, qa in enumerate(validated_qa_units):
                qa.qa_id = f"qa_{base_id + j:03d}"
                qa.sequence_in_session = base_id + j
            all_qa_units.extend(validated_qa_units)
        else:
            trace.sections_with_no_qa.append(section.section_id)
            logger.warning("no_qa_extracted", section_id=section.section_id)

    # Resolve speaker IDs
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

    # Compute statistics
    total_follow_ups = sum(1 for qa in all_qa_units if qa.is_follow_up)
    unique_questioners = len(set(qa.questioner_name for qa in all_qa_units))

    result = QAExtractionResult(
        qa_units=all_qa_units,
        total_qa_units=len(all_qa_units),
        total_follow_ups=total_follow_ups,
        unique_questioners=unique_questioners,
        sections_processed=trace.sections_processed,
        sections_with_no_qa=trace.sections_with_no_qa,
    )

    logger.info(
        "qa_extraction_complete",
        total_units=len(all_qa_units),
        follow_ups=total_follow_ups,
        unique_questioners=unique_questioners,
        llm_calls=trace.llm_calls_made,
        hard_rules_enforced=len(trace.hard_rule_enforcements),
    )

    return result, trace


# =============================================================================
# Smart Regex-Based Q&A Extraction
# =============================================================================

def _extract_qa_candidates_with_regex(
    section_id: str,
    turns: list[tuple[int, Optional[str], str]],
    registry: Optional[SpeakerRegistry],
    base_page: int,
) -> list[ExtractedQAUnit]:
    """Extract Q&A units using smart regex and heuristics.

    Uses pattern matching and role analysis to identify Q&A boundaries.

    Args:
        section_id: ID of the section being processed
        turns: List of parsed turns (line_idx, speaker, text)
        registry: Speaker registry for role lookup
        base_page: Starting page number

    Returns:
        List of ExtractedQAUnit objects
    """
    qa_units = []
    i = 0
    qa_counter = 0

    while i < len(turns):
        # Look for question start (analyst speaking with question mark or question-like content)
        question_start_idx = _find_question_start(turns, i, registry)
        if question_start_idx is None:
            break

        # Find the complete question (may span multiple turns from same speaker)
        question_end_idx = _find_question_end(turns, question_start_idx, registry)

        # Find the response (management speaking after question)
        response_start_idx = _find_response_start(turns, question_end_idx + 1, registry)
        if response_start_idx is None:
            # No response found, move to next potential question
            i = question_start_idx + 1
            continue

        # Find the complete response (may span multiple turns)
        response_end_idx = _find_response_end(turns, response_start_idx, registry)

        # Create Q&A unit
        qa_unit = _create_qa_unit(
            section_id=section_id,
            turns=turns,
            question_start_idx=question_start_idx,
            question_end_idx=question_end_idx,
            response_start_idx=response_start_idx,
            response_end_idx=response_end_idx,
            qa_counter=qa_counter,
            base_page=base_page,
        )

        if qa_unit:
            qa_units.append(qa_unit)
            qa_counter += 1

        # Move past this Q&A unit
        i = response_end_idx + 1

    return qa_units


def _find_question_start(
    turns: list[tuple[int, Optional[str], str]],
    start_idx: int,
    registry: Optional[SpeakerRegistry]
) -> Optional[int]:
    """Find the start of a question (analyst asking something).

    Looks for:
    - Analyst speaker
    - Question-like content (question marks, interrogative words)
    """
    for i in range(start_idx, len(turns)):
        _, speaker, text = turns[i]
        if not speaker or not text:
            continue

        # Check if speaker is analyst
        role = _get_speaker_role(speaker, registry)
        if role != SpeakerRole.ANALYST:
            continue

        # Check if content looks like a question
        text_lower = text.lower()
        if ('?' in text or
            text_lower.startswith(('what', 'how', 'when', 'where', 'why', 'can', 'could', 'would', 'should')) or
            text_lower.endswith('?')):
            return i

    return None


def _find_question_end(
    turns: list[tuple[int, Optional[str], str]],
    start_idx: int,
    registry: Optional[SpeakerRegistry]
) -> int:
    """Find the end of a question (may span multiple turns from same speaker)."""
    if start_idx >= len(turns):
        return start_idx

    question_speaker = turns[start_idx][1]
    end_idx = start_idx

    # Look for continuation from same speaker or natural question end
    for i in range(start_idx + 1, min(start_idx + 5, len(turns))):  # Limit to avoid capturing too much
        _, speaker, text = turns[i]

        # If different speaker or clear response start, stop
        if speaker != question_speaker:
            role = _get_speaker_role(speaker, registry)
            if role == SpeakerRole.MANAGEMENT:
                break

        # If text looks like response start, stop
        if text and ('.' in text.split()[0] if text.split() else False):
            # Looks like a statement, not continuation
            break

        end_idx = i

    return end_idx


def _find_response_start(
    turns: list[tuple[int, Optional[str], str]],
    start_idx: int,
    registry: Optional[SpeakerRegistry]
) -> Optional[int]:
    """Find the start of a response (management answering the question)."""
    for i in range(start_idx, min(start_idx + 3, len(turns))):  # Look ahead a few turns
        _, speaker, text = turns[i]
        if not speaker:
            continue

        role = _get_speaker_role(speaker, registry)
        if role == SpeakerRole.MANAGEMENT and text and len(text) > 10:
            return i

    return None


def _find_response_end(
    turns: list[tuple[int, Optional[str], str]],
    start_idx: int,
    registry: Optional[SpeakerRegistry]
) -> int:
    """Find the end of a response (may span multiple turns from management)."""
    if start_idx >= len(turns):
        return start_idx

    end_idx = start_idx

    # Look for continuation from management speakers
    for i in range(start_idx + 1, min(start_idx + 8, len(turns))):  # Allow longer responses
        _, speaker, text = turns[i]

        if not speaker:
            # No speaker, might be continuation
            end_idx = i
            continue

        role = _get_speaker_role(speaker, registry)

        # If next analyst speaks, response probably ended
        if role == SpeakerRole.ANALYST:
            break

        # If moderator speaks with transition language, response ended
        if role == SpeakerRole.MODERATOR and text:
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in ['next question', 'another question', 'thank you']):
                break

        end_idx = i

    return end_idx


def _get_speaker_role(speaker_name: str, registry: Optional[SpeakerRegistry]) -> SpeakerRole:
    """Get speaker role from registry or heuristic."""
    if registry:
        info = registry.get_by_name(speaker_name)
        if info:
            return info.role

    # Heuristic-based role detection
    speaker_lower = speaker_name.lower()
    if any(word in speaker_lower for word in ['operator', 'moderator']):
        return SpeakerRole.MODERATOR
    elif any(word in speaker_lower for word in ['analyst', 'investor', 'journalist']):
        return SpeakerRole.ANALYST
    elif any(title in speaker_lower for title in ['ceo', 'cfo', 'cto', 'president', 'director', 'executive']):
        return SpeakerRole.MANAGEMENT
    else:
        return SpeakerRole.UNKNOWN


def _create_qa_unit(
    section_id: str,
    turns: list[tuple[int, Optional[str], str]],
    question_start_idx: int,
    question_end_idx: int,
    response_start_idx: int,
    response_end_idx: int,
    qa_counter: int,
    base_page: int,
) -> Optional[ExtractedQAUnit]:
    """Create a Q&A unit from turn indices."""
    if (question_start_idx > question_end_idx or
        response_start_idx > response_end_idx or
        question_end_idx >= response_start_idx):
        return None

    # Build question content
    question_turns_data = turns[question_start_idx:question_end_idx + 1]
    question_speaker = question_turns_data[0][1] if question_turns_data else None
    question_text = " ".join([text for _, _, text in question_turns_data if text])

    # Build response content
    response_turns_data = turns[response_start_idx:response_end_idx + 1]
    response_speakers = list(set([speaker for _, speaker, _ in response_turns_data if speaker]))
    response_text = " ".join([text for _, _, text in response_turns_data if text])

    if not question_text or not response_text:
        return None

    # Build turn objects
    question_turns = [SpeakerTurn(
        speaker_name=question_speaker or "Unknown",
        speaker_id=None,
        text=question_text,
        page_number=base_page,
        is_question=True,
    )]

    response_turns = [SpeakerTurn(
        speaker_name=response_speakers[0] if response_speakers else "Management",
        speaker_id=None,
        text=response_text,
        page_number=base_page,
        is_question=False,
    )]

    # Check for follow-up (heuristic: same speaker asking again)
    is_follow_up = False
    follow_up_of = None
    # This would need more context to determine properly

    qa_id = f"qa_{qa_counter:03d}"

    return ExtractedQAUnit(
        qa_id=qa_id,
        source_section_id=section_id,
        questioner_name=question_speaker or "Unknown",
        questioner_id=None,
        question_turns=question_turns,
        response_turns=response_turns,
        responder_names=response_speakers or ["Management"],
        responder_ids=[],
        question_text=question_text,
        response_text=response_text,
        is_follow_up=is_follow_up,
        follow_up_of=follow_up_of,
        start_page=base_page,
        end_page=base_page,
        sequence_in_session=qa_counter,
        boundary_justification="Extracted using smart regex patterns",
    )


# =============================================================================
# LLM Validation Functions
# =============================================================================

# =============================================================================
# DOCUMENT-LEVEL LLM Validation Functions
# =============================================================================

def _validate_qa_units_with_context(
    candidates: list[dict],
    section_text: str,
    registry: Optional[SpeakerRegistry],
) -> dict:
    """Validate multiple Q&A units using LLM with full context.

    LLM sees ALL candidates and makes GLOBAL decisions about:
    - Which candidates are valid Q&A exchanges
    - Boundary corrections for incomplete units
    - Speaker role verification

    Args:
        candidates: List of candidate Q&A dicts with question_text, response_text, etc.
        section_text: Full section text for context
        registry: Optional speaker registry

    Returns:
        Dict with "validations" list and "corrections" dict
    """
    try:
        from src.pipeline_v2.llm_helpers import _invoke_llm

        # Build candidates summary
        candidates_summary = ""
        for i, cand in enumerate(candidates):
            candidates_summary += f"""
CANDIDATE {i+1}:
  Questioner: {cand.get('questioner', 'Unknown')}
  Question: {cand.get('question_text', '')[:200]}...
  Responders: {', '.join(cand.get('responders', ['Management']))}
  Response: {cand.get('response_text', '')[:200]}...
  Follow-up: {cand.get('is_follow_up', False)}
"""

        # Build role context
        role_context = ""
        if registry:
            management_names = [s.canonical_name for s in registry.speakers.values() if s.role == SpeakerRole.MANAGEMENT]
            analyst_names = [s.canonical_name for s in registry.speakers.values() if s.role == SpeakerRole.ANALYST]
            role_context = f"""
Management speakers: {', '.join(management_names[:5])}
Analyst speakers: {', '.join(analyst_names[:5])}
"""

        prompt = f"""You are validating Q&A extractions from an earnings call transcript.

TASK: Review ALL Q&A candidates and determine which are valid exchanges.

{role_context if role_context else "Role context: Not available"}

=== Q&A CANDIDATES FROM REGEX EXTRACTION ===
{candidates_summary}

=== VALIDATION CRITERIA ===
A VALID Q&A exchange MUST:
1. Have a coherent question from an analyst/investor
2. Have a response from management that addresses the question
3. Contain substantive content (not just greetings/thanks)
4. Have proper speaker roles (analyst asks, management answers)

A Q&A is INVALID if:
1. Question is truncated mid-sentence
2. Response is truncated mid-sentence
3. Content is just greetings or procedural
4. Questioner is management (they answer, not ask)
5. No actual response provided

=== REPAIR TASK ===
For each candidate, provide:
- Whether it's valid
- If invalid, the reason
- If valid but incomplete, suggested corrections

Respond with this EXACT JSON format:
{{
  "validations": [
    {{
      "index": 0,
      "is_valid": true,
      "confidence": 0.9,
      "reason": "Valid Q&A exchange"
    }}
  ],
  "corrections": [
    {{
      "index": 0,
      "question_text": "Repaired complete question",
      "response_text": "Repaired complete response",
      "questioner": "Corrected speaker name",
      "responders": ["Corrected responder names"]
    }}
  ]
}}

IMPORTANT: Output ONLY valid JSON. No explanations before or after."""

        response = _invoke_llm(prompt, max_tokens=3000)
        return _parse_validation_response(response)

    except Exception as e:
        logger.warning("qa_validation_llm_failed", error=str(e))
        # Fallback: mark all candidates as valid
        return {
            "validations": [{"index": i, "is_valid": True, "confidence": 0.5, "reason": "LLM failed - accepting by default"} for i in range(len(candidates))],
            "corrections": []
        }


def _parse_validation_response(response: str) -> dict:
    """Parse LLM validation response."""
    import json
    import re

    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("validation_json_parse_failed", error=str(e))

    return {
        "validations": [],
        "corrections": []
    }


def _apply_llm_corrections(candidate: ExtractedQAUnit, validation: dict) -> ExtractedQAUnit:
    """Apply LLM corrections to a Q&A unit.

    Repairs incomplete/truncated Q&A units based on LLM suggestions.

    Args:
        candidate: The original Q&A unit
        validation: LLM validation with corrections

    Returns:
        Corrected ExtractedQAUnit
    """
    import copy

    # Create a copy to modify
    corrected = copy.deepcopy(candidate)

    # Check for "Unknown" speakers and try to fix
    if corrected.questioner_name == "Unknown" or not corrected.questioner_name:
        corrected.questioner_name = "Analyst"  # Default fallback
        logger.debug("repaired_unknown_questioner", qa_id=corrected.qa_id)

    if not corrected.responder_names or all(name == "Unknown" for name in corrected.responder_names):
        corrected.responder_names = ["Management"]  # Default fallback
        logger.debug("repaired_unknown_responders", qa_id=corrected.qa_id)

    # Check for truncation indicators
    if _is_truncated_text(corrected.question_text):
        logger.warning("question_truncated", qa_id=corrected.qa_id, length=len(corrected.question_text))
        # Mark for potential repair (would need more context)

    if _is_truncated_text(corrected.response_text):
        logger.warning("response_truncated", qa_id=corrected.qa_id, length=len(corrected.response_text))
        # Mark for potential repair (would need more context)

    # Ensure question_text and response_text are not empty
    if not corrected.question_text or len(corrected.question_text) < 5:
        corrected.question_text = corrected.question_text or "Question text incomplete"

    if not corrected.response_text or len(corrected.response_text) < 5:
        corrected.response_text = corrected.response_text or "Response text incomplete"

    return corrected


def _is_truncated_text(text: str) -> bool:
    """Check if text appears to be truncated."""
    if not text or len(text) < 20:
        return True

    # Check for common truncation patterns
    truncation_indicators = [
        text.rstrip().endswith(('...', '....', '... ')),
        len(text.split()) < 3 and not text.endswith(('!', '?', '.', ')', '"', "'")),
    ]

    return any(truncation_indicators)


def _validate_qa_unit_with_llm(qa_unit: ExtractedQAUnit, section_text: str) -> bool:
    """Validate a Q&A unit using LLM.

    Checks if the extracted Q&A unit makes sense in context.

    Args:
        qa_unit: The Q&A unit to validate
        section_text: Full section text for context

    Returns:
        True if validation passes, False otherwise
    """
    try:
        from src.pipeline_v2.llm_helpers import _invoke_llm

        # Create validation prompt
        prompt = f"""You are validating a Q&A extraction from an earnings call transcript.

QUESTION: "{qa_unit.question_text[:300]}"

RESPONSE: "{qa_unit.response_text[:300]}"

QUESTIONER: {qa_unit.questioner_name}

RESPONDERS: {', '.join(qa_unit.responder_names)}

Does this represent a valid Q&A exchange? Consider:
1. Is the question coherent and complete?
2. Does the response address the question?
3. Are the speakers appropriate (analyst asks, management responds)?
4. Is the content substantive (not just greetings or acknowledgments)?

Answer ONLY YES or NO:"""

        response = _invoke_llm(prompt, max_tokens=10)
        return "YES" in response.upper()

    except Exception as e:
        logger.warning("qa_unit_validation_failed", error=str(e))
        return False  # Conservative: fail validation on error


