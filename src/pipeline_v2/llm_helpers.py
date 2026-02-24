"""LLM Helper Functions for Hybrid Intelligence Pipeline.

This module provides constrained LLM prompts for the hybrid pipeline.
All LLM calls are designed to:
- Return binary (YES/NO) or choice-based responses
- Be schema-validated
- Assist decisions, not control the pipeline

Never use open-ended prompts like "segment this transcript".
"""

import json
import re
from typing import Literal, Optional

import structlog
from pydantic import BaseModel, Field

from src.llm.client import create_json_llm_client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = structlog.get_logger(__name__)


# =============================================================================
# Response Models (Schema-Validated)
# =============================================================================

class BinaryDecision(BaseModel):
    """Binary YES/NO decision with reasoning."""
    decision: Literal["YES", "NO"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=200)


class RoleClassification(BaseModel):
    """Speaker role classification."""
    role: Literal["management", "analyst", "moderator", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=200)


class TurnClassification(BaseModel):
    """Turn classification (question vs answer)."""
    turn_type: Literal["question", "answer", "transition", "other"]
    confidence: float = Field(ge=0.0, le=1.0)


class NameNormalization(BaseModel):
    """Normalized speaker name."""
    canonical_name: str = Field(max_length=50)
    is_valid_person_name: bool
    confidence: float = Field(ge=0.0, le=1.0)


class AliasMergeDecision(BaseModel):
    """Decision on whether two names refer to same person."""
    same_person: Literal["YES", "NO"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=200)


# =============================================================================
# ENHANCED DECISION MODELS (Contextual Decision Power)
# =============================================================================

class EvidenceSpan(BaseModel):
    """A text span that supports a decision."""
    text: str = Field(max_length=300, description="The supporting text")
    source: str = Field(max_length=100, description="Where this text came from (e.g., 'turn_3', 'introduction')")
    relevance: str = Field(max_length=100, description="Why this evidence matters")


class SpeakerVerificationDecision(BaseModel):
    """Comprehensive LLM decision for speaker verification.

    This model gives the LLM contextual decision power to:
    - Reject fake speakers (sentence fragments, content text)
    - Correct roles based on full context
    - Verify or reject title assignments
    - Prevent incorrect alias merges
    """
    # Core identity verification
    is_real_person: bool = Field(description="Is this a valid human speaker name?")
    canonical_name: str = Field(max_length=50, description="Normalized name (FirstName LastName)")

    # Role classification with evidence
    role: Literal["moderator", "management", "analyst", "unknown"] = Field(
        description="Speaker role based on full context"
    )
    role_confidence: float = Field(ge=0.0, le=1.0)

    # Title verification (ONLY if explicitly stated in transcript)
    title: Optional[str] = Field(
        None, max_length=50,
        description="Job title ONLY if explicitly stated in text (e.g., 'CEO', 'CFO')"
    )
    title_verified: bool = Field(
        default=False,
        description="True only if title appears explicitly in transcript"
    )

    # Company affiliation (for analysts)
    company: Optional[str] = Field(
        None, max_length=50,
        description="Company affiliation if mentioned"
    )

    # Alias merge decision
    merge_with: Optional[str] = Field(
        None,
        description="speaker_id to merge with, if this is an alias"
    )
    merge_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Evidence and reasoning
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Text snippets justifying this decision"
    )
    reasoning: str = Field(max_length=300, description="Explanation of decision")

    # Rejection reasons (if is_real_person=False)
    rejection_reason: Optional[str] = Field(
        None, max_length=200,
        description="Why this was rejected as a speaker"
    )


class TurnIntentDecision(BaseModel):
    """LLM decision for classifying turn intent in Q&A segmentation.

    This model governs the Q&A state machine:
    - Question ends ONLY when LLM says QUESTION_END
    - Answer starts ONLY when LLM says ANSWER_START
    - Moderator speech is always MODERATOR_TRANSITION
    """
    intent: Literal[
        "QUESTION_START",       # Beginning of a new question
        "QUESTION_CONTINUATION", # Continuation of current question
        "QUESTION_END",          # End of question (usually implicit)
        "ANSWER_START",          # Beginning of management response
        "ANSWER_CONTINUATION",   # Continuation of current answer
        "MODERATOR_TRANSITION",  # Moderator introducing next Q&A
        "NON_QA"                 # Not part of Q&A (e.g., technical issues)
    ] = Field(description="The semantic intent of this turn")

    # Does this turn support Q&A extraction?
    supports_qa: bool = Field(
        description="True if this turn contributes to a Q&A unit"
    )

    # Confidence
    confidence: float = Field(ge=0.0, le=1.0)

    # Evidence supporting the classification
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Text snippets justifying intent classification"
    )

    # For linking follow-ups
    is_follow_up: bool = Field(
        default=False,
        description="True if this continues a previous question's topic"
    )
    follow_up_reason: Optional[str] = Field(
        None, max_length=200,
        description="Why this is a follow-up"
    )


# =============================================================================
# LLM Invocation Helpers
# =============================================================================

def _invoke_llm(prompt: str, max_tokens: int = 200) -> str:
    """Invoke LLM with a prompt and return raw response."""
    llm = create_json_llm_client()
    try:
        response = llm.invoke(prompt)
        return response.strip() if response else ""
    except Exception as e:
        logger.error("llm_invocation_failed", error=str(e))
        return ""


def _parse_binary_response(response: str) -> tuple[bool, float, str]:
    """Parse a YES/NO response from LLM.

    Returns:
        Tuple of (is_yes, confidence, reasoning)
    """
    response_upper = response.upper()

    # Try to find YES or NO
    if "YES" in response_upper:
        decision = True
    elif "NO" in response_upper:
        decision = False
    else:
        # Default to NO if unclear
        decision = False

    # Try to extract confidence
    confidence = 0.7  # Default
    conf_match = re.search(r"confidence[:\s]*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            if confidence > 1:
                confidence = confidence / 100
        except ValueError:
            pass

    # Extract reasoning
    reasoning = response[:200] if response else "No reasoning provided"

    return decision, confidence, reasoning


def _parse_choice_response(response: str, valid_choices: list[str]) -> tuple[str, float]:
    """Parse a choice response from LLM.

    Returns:
        Tuple of (choice, confidence)
    """
    response_lower = response.lower()

    for choice in valid_choices:
        if choice.lower() in response_lower:
            # Try to extract confidence
            confidence = 0.7
            conf_match = re.search(r"confidence[:\s]*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    if confidence > 1:
                        confidence = confidence / 100
                except ValueError:
                    pass
            return choice, confidence

    # Default to first choice if none found
    return valid_choices[0], 0.3


# =============================================================================
# Boundary Detection Helpers
# =============================================================================

def confirm_qa_session_start(text_snippet: str, context_before: str = "") -> BinaryDecision:
    """Ask LLM to confirm if text indicates start of Q&A session.

    Args:
        text_snippet: The candidate boundary text (50-100 chars)
        context_before: Previous context for reference

    Returns:
        BinaryDecision with YES/NO
    """
    prompt = f"""You are analyzing an earnings call transcript.

Context before: {context_before[-200:] if context_before else "START OF DOCUMENT"}

Candidate text: "{text_snippet}"

Does this text indicate the START of a Q&A (Questions and Answers) session?

Indicators of Q&A start:
- Moderator says "we will now take questions"
- "Let's open the floor for questions"
- "First question comes from..."
- Transition from prepared remarks to interactive session

Answer ONLY with:
YES - if this clearly starts a Q&A session
NO - if this is NOT a Q&A session start

Decision:"""

    response = _invoke_llm(prompt)
    is_yes, confidence, reasoning = _parse_binary_response(response)

    return BinaryDecision(
        decision="YES" if is_yes else "NO",
        confidence=confidence,
        reasoning=reasoning[:200]
    )


def confirm_section_boundary(text_snippet: str, current_section: str) -> BinaryDecision:
    """Ask LLM to confirm if text indicates a section transition.

    Args:
        text_snippet: The candidate boundary text
        current_section: Current section type (opening_remarks, qa_session, etc.)

    Returns:
        BinaryDecision
    """
    prompt = f"""You are analyzing an earnings call transcript.

Current section: {current_section}
Candidate text: "{text_snippet[:150]}"

Does this text indicate a TRANSITION to a different section?

Section types:
- opening_remarks: Prepared statements by executives
- qa_session: Questions from analysts, answers from management
- closing_remarks: Final wrap-up statements

Answer ONLY with:
YES - if this clearly transitions to a different section
NO - if this continues the current section

Decision:"""

    response = _invoke_llm(prompt)
    is_yes, confidence, reasoning = _parse_binary_response(response)

    return BinaryDecision(
        decision="YES" if is_yes else "NO",
        confidence=confidence,
        reasoning=reasoning[:200]
    )


# =============================================================================
# ENHANCED SPEAKER VERIFICATION (Contextual Decision Power)
# =============================================================================

def verify_speaker_with_context(
    candidate_name: str,
    all_turns_by_speaker: list[str],
    moderator_introductions: list[str],
    opening_remarks_context: str,
    existing_registry_snapshot: list[dict],
    detected_aliases: list[tuple[str, float]],
) -> SpeakerVerificationDecision:
    """Verify a speaker candidate with full document context.

    This gives the LLM contextual decision power to:
    - Reject fake speakers (sentence fragments)
    - Correct roles based on full context
    - Verify titles only if explicitly stated
    - Prevent incorrect alias merges

    Args:
        candidate_name: The name to verify
        all_turns_by_speaker: All text spoken by this speaker
        moderator_introductions: Lines where moderator introduced this speaker
        opening_remarks_context: Opening remarks for title/role hints
        existing_registry_snapshot: Current speakers already registered
        detected_aliases: (alias_name, similarity_score) pairs

    Returns:
        SpeakerVerificationDecision with full verification result
    """
    # Build context for LLM
    turns_sample = "\n".join(all_turns_by_speaker[:5])[:1000]  # First 5 turns, max 1000 chars
    intros = "\n".join(moderator_introductions[:3])[:500]
    registry_names = [s.get("canonical_name", "") for s in existing_registry_snapshot[:10]]
    alias_info = ", ".join([f"{a[0]} ({a[1]:.0%})" for a in detected_aliases[:3]])

    prompt = f"""You are verifying speaker identity in an earnings call transcript.

CANDIDATE NAME: "{candidate_name}"

TURNS SPOKEN BY THIS SPEAKER:
{turns_sample}

MODERATOR INTRODUCTIONS MENTIONING THIS SPEAKER:
{intros if intros else "None found"}

OPENING REMARKS CONTEXT:
{opening_remarks_context[:500] if opening_remarks_context else "Not available"}

ALREADY REGISTERED SPEAKERS:
{", ".join(registry_names) if registry_names else "None yet"}

POTENTIAL ALIASES (with similarity):
{alias_info if alias_info else "None detected"}

VERIFICATION TASKS:
1. Is this a REAL PERSON name? (Reject: "Conference Call", "Before we begin", sentences)
2. What is their ROLE? (moderator/management/analyst/unknown)
3. What is their TITLE? (ONLY if explicitly stated, e.g., "John Smith, CFO")
4. Should this merge with an existing speaker?

HARD RULES:
- "Moderator" or "Operator" is always role=moderator with NO title
- Titles (CEO, CFO, etc.) require EXPLICIT text evidence
- Names with >4 words or containing verbs are likely NOT real names
- Different first names (Tarak Patel vs Shanti Patel) are DIFFERENT people

Respond in this EXACT format:
IS_REAL_PERSON: [YES/NO]
CANONICAL_NAME: [normalized name or INVALID]
ROLE: [moderator/management/analyst/unknown]
ROLE_CONFIDENCE: [0.0-1.0]
TITLE: [title or NONE]
TITLE_VERIFIED: [YES/NO - YES only if title appears explicitly]
COMPANY: [company name or NONE]
MERGE_WITH: [existing speaker name or NONE]
MERGE_CONFIDENCE: [0.0-1.0]
EVIDENCE: [quote supporting your decision]
REASONING: [brief explanation]
REJECTION_REASON: [if IS_REAL_PERSON=NO, explain why]"""

    response = _invoke_llm(prompt, max_tokens=500)

    # Parse response
    result = _parse_speaker_verification_response(response, candidate_name)
    return result


def _parse_speaker_verification_response(
    response: str,
    fallback_name: str
) -> SpeakerVerificationDecision:
    """Parse LLM response into SpeakerVerificationDecision."""
    lines = response.strip().split('\n')

    # Defaults
    is_real = False
    canonical_name = fallback_name
    role = "unknown"
    role_conf = 0.5
    title = None
    title_verified = False
    company = None
    merge_with = None
    merge_conf = 0.0
    evidence_text = ""
    reasoning = ""
    rejection = None

    for line in lines:
        upper_line = line.upper()
        if 'IS_REAL_PERSON:' in upper_line:
            is_real = 'YES' in upper_line
        elif 'CANONICAL_NAME:' in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('INVALID', 'NONE', ''):
                canonical_name = val[:50]
        elif 'ROLE:' in upper_line and 'ROLE_CONFIDENCE' not in upper_line:
            val = line.split(':', 1)[-1].strip().lower()
            if val in ('moderator', 'management', 'analyst', 'unknown'):
                role = val
        elif 'ROLE_CONFIDENCE:' in upper_line:
            try:
                role_conf = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass
        elif 'TITLE:' in upper_line and 'TITLE_VERIFIED' not in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('NONE', 'N/A', ''):
                title = val[:50]
        elif 'TITLE_VERIFIED:' in upper_line:
            title_verified = 'YES' in upper_line
        elif 'COMPANY:' in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('NONE', 'N/A', ''):
                company = val[:50]
        elif 'MERGE_WITH:' in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('NONE', 'N/A', ''):
                merge_with = val
        elif 'MERGE_CONFIDENCE:' in upper_line:
            try:
                merge_conf = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass
        elif 'EVIDENCE:' in upper_line:
            evidence_text = line.split(':', 1)[-1].strip()[:300]
        elif 'REASONING:' in upper_line:
            reasoning = line.split(':', 1)[-1].strip()[:300]
        elif 'REJECTION_REASON:' in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('NONE', 'N/A', ''):
                rejection = val[:200]

    # Build evidence spans
    evidence_spans = []
    if evidence_text:
        evidence_spans.append(EvidenceSpan(
            text=evidence_text,
            source="llm_verification",
            relevance="Primary evidence for verification decision"
        ))

    # If not verified as real and no rejection reason, add one
    if not is_real and not rejection:
        rejection = "LLM determined this is not a valid speaker name"

    return SpeakerVerificationDecision(
        is_real_person=is_real,
        canonical_name=canonical_name,
        role=role,
        role_confidence=role_conf,
        title=title if title_verified else None,
        title_verified=title_verified,
        company=company,
        merge_with=merge_with,
        merge_confidence=merge_conf,
        evidence_spans=evidence_spans,
        reasoning=reasoning,
        rejection_reason=rejection
    )


# =============================================================================
# Speaker Registry Helpers (Legacy - kept for compatibility)
# =============================================================================

def normalize_speaker_name(raw_name: str, context: str = "") -> NameNormalization:
    """Ask LLM to normalize a speaker name.

    Args:
        raw_name: The raw extracted name
        context: Surrounding context

    Returns:
        NameNormalization with canonical name
    """
    prompt = f"""You are extracting speaker names from an earnings call transcript.

Raw extracted text: "{raw_name}"
Context: "{context[:200]}"

Task: Determine if this is a valid PERSON NAME and normalize it.

Rules:
- Valid: "John Smith", "Mr. Tarak Patel", "JANE DOE"
- Invalid: "Conference Call", "Before we begin", "The next question"
- Remove honorifics (Mr., Ms., Dr.) for canonical name
- Canonical name should be "FirstName LastName" format

Respond in this exact format:
CANONICAL_NAME: [the normalized name or INVALID]
IS_VALID: [YES or NO]
CONFIDENCE: [0.0 to 1.0]"""

    response = _invoke_llm(prompt)

    # Parse response
    canonical_name = raw_name
    is_valid = False
    confidence = 0.5

    lines = response.strip().split('\n')
    for line in lines:
        if 'CANONICAL_NAME:' in line.upper():
            name = line.split(':', 1)[-1].strip()
            if name.upper() != 'INVALID':
                canonical_name = name
                is_valid = True
        elif 'IS_VALID:' in line.upper():
            is_valid = 'YES' in line.upper()
        elif 'CONFIDENCE:' in line.upper():
            try:
                confidence = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass

    return NameNormalization(
        canonical_name=canonical_name[:50],
        is_valid_person_name=is_valid,
        confidence=confidence
    )


def classify_speaker_role(
    speaker_name: str,
    context: str,
    is_questioner: bool = False,
    has_title: Optional[str] = None
) -> RoleClassification:
    """Ask LLM to classify speaker role.

    Args:
        speaker_name: The speaker's name
        context: Surrounding transcript context
        is_questioner: Whether they asked a question
        has_title: Extracted title if any

    Returns:
        RoleClassification
    """
    prompt = f"""You are classifying speakers in an earnings call transcript.

Speaker: "{speaker_name}"
Title (if known): {has_title or "Unknown"}
Asked a question: {"Yes" if is_questioner else "No"}
Context: "{context[:300]}"

Classify this speaker's role:

MODERATOR: Conference call operator, IR contact who introduces speakers
MANAGEMENT: Company executives (CEO, CFO, etc.) who present and answer questions
ANALYST: Investment analysts who ASK questions about the company
UNKNOWN: Cannot determine

Rules:
- "Moderator" or "Operator" as name → MODERATOR
- Introduced by "question from [NAME]" → ANALYST
- Has executive title (CEO, CFO, MD, etc.) → MANAGEMENT
- Answers questions → MANAGEMENT

Respond in this exact format:
ROLE: [moderator/management/analyst/unknown]
CONFIDENCE: [0.0 to 1.0]
REASONING: [brief explanation]"""

    response = _invoke_llm(prompt)

    # Parse response
    role = "unknown"
    confidence = 0.5
    reasoning = ""

    valid_roles = ["moderator", "management", "analyst", "unknown"]

    lines = response.strip().split('\n')
    for line in lines:
        if 'ROLE:' in line.upper():
            role_str = line.split(':', 1)[-1].strip().lower()
            if role_str in valid_roles:
                role = role_str
        elif 'CONFIDENCE:' in line.upper():
            try:
                confidence = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass
        elif 'REASONING:' in line.upper():
            reasoning = line.split(':', 1)[-1].strip()

    return RoleClassification(
        role=role,
        confidence=confidence,
        reasoning=reasoning[:200]
    )


def should_merge_names(name1: str, name2: str, context1: str = "", context2: str = "") -> AliasMergeDecision:
    """Ask LLM if two names refer to the same person.

    Args:
        name1: First name
        name2: Second name
        context1: Context where name1 appeared
        context2: Context where name2 appeared

    Returns:
        AliasMergeDecision
    """
    prompt = f"""You are deduplicating speaker names in an earnings call transcript.

Name 1: "{name1}"
Context 1: "{context1[:150]}"

Name 2: "{name2}"
Context 2: "{context2[:150]}"

Do these names refer to the SAME PERSON?

Rules:
- "Mr. John Smith" and "John Smith" → SAME PERSON
- "Tarak Patel" and "Shanti Patel" → DIFFERENT PEOPLE (different first names)
- "CEO" and "John Smith, CEO" → SAME if context confirms
- Similar but different spellings need careful evaluation

Answer ONLY with:
YES - same person
NO - different people

Decision:"""

    response = _invoke_llm(prompt)
    is_yes, confidence, reasoning = _parse_binary_response(response)

    return AliasMergeDecision(
        same_person="YES" if is_yes else "NO",
        confidence=confidence,
        reasoning=reasoning[:200]
    )


# =============================================================================
# ENHANCED Q&A EXTRACTION (LLM-Governed State Machine)
# =============================================================================

def classify_turn_intent(
    current_turn_text: str,
    current_speaker: str,
    current_speaker_role: str,
    previous_turns: list[tuple[str, str, str]],  # (speaker, role, text)
    next_turn: Optional[tuple[str, str, str]],   # (speaker, role, text)
    current_qa_state: str,  # "idle", "in_question", "in_answer"
    current_question_text: str = "",
) -> TurnIntentDecision:
    """Classify turn intent for Q&A segmentation with full context.

    This LLM decision GOVERNS the Q&A state machine:
    - Question ends ONLY when LLM says QUESTION_END or ANSWER_START
    - Answer starts ONLY when LLM says ANSWER_START
    - Moderator speech is ALWAYS classified as MODERATOR_TRANSITION

    Args:
        current_turn_text: The text of the current turn
        current_speaker: Name of current speaker
        current_speaker_role: Role (moderator/management/analyst/unknown)
        previous_turns: Last 2 turns as (speaker, role, text) tuples
        next_turn: Next turn if available
        current_qa_state: Current state machine state
        current_question_text: Accumulated question text if in_question state

    Returns:
        TurnIntentDecision governing the state transition
    """
    # HARD RULE: Moderator is ALWAYS transition
    if current_speaker_role == "moderator":
        return TurnIntentDecision(
            intent="MODERATOR_TRANSITION",
            supports_qa=False,
            confidence=0.99,
            evidence_spans=[EvidenceSpan(
                text=current_turn_text[:100],
                source="speaker_role",
                relevance="Speaker is moderator - always transition"
            )],
            is_follow_up=False
        )

    # Build context for LLM
    prev_context = ""
    for i, (spk, role, txt) in enumerate(previous_turns[-2:]):
        prev_context += f"[Turn -{2-i}] {spk} ({role}): {txt[:200]}\n"

    next_context = ""
    if next_turn:
        spk, role, txt = next_turn
        next_context = f"[Next Turn] {spk} ({role}): {txt[:200]}"

    state_desc = {
        "idle": "No active Q&A - waiting for a question",
        "in_question": f"Currently collecting question from analyst. Question so far: '{current_question_text[:200]}'",
        "in_answer": "Currently collecting answer from management"
    }.get(current_qa_state, current_qa_state)

    prompt = f"""You are segmenting an earnings call Q&A session into question-answer pairs.

CURRENT STATE: {state_desc}

PREVIOUS TURNS:
{prev_context if prev_context else "None (start of Q&A)"}

CURRENT TURN:
Speaker: {current_speaker}
Role: {current_speaker_role}
Text: "{current_turn_text[:500]}"

NEXT TURN PREVIEW:
{next_context if next_context else "Unknown / end of transcript"}

CLASSIFICATION TASK:
What is the INTENT of the current turn?

POSSIBLE INTENTS:
- QUESTION_START: This turn BEGINS a new question from an analyst
- QUESTION_CONTINUATION: This turn CONTINUES an ongoing question
- ANSWER_START: This turn BEGINS a response from management
- ANSWER_CONTINUATION: This turn CONTINUES an ongoing response
- MODERATOR_TRANSITION: This is moderator speech (introducing next Q&A)
- NON_QA: This is not part of Q&A (technical issues, greetings, etc.)

RULES:
1. Analysts ASK questions (QUESTION_START/CONTINUATION)
2. Management ANSWERS questions (ANSWER_START/CONTINUATION)
3. A QUESTION ends when management starts answering
4. Questions with "?" likely QUESTION_START or CONTINUATION
5. "Thank you" or "Good morning" alone is often a preamble, not the question itself
6. If analyst speaks after an answer, it's likely a FOLLOW-UP QUESTION

Respond in this EXACT format:
INTENT: [QUESTION_START/QUESTION_CONTINUATION/ANSWER_START/ANSWER_CONTINUATION/MODERATOR_TRANSITION/NON_QA]
SUPPORTS_QA: [YES/NO]
CONFIDENCE: [0.0-1.0]
IS_FOLLOW_UP: [YES/NO]
FOLLOW_UP_REASON: [reason if follow-up, else NONE]
EVIDENCE: [key phrase supporting your classification]"""

    response = _invoke_llm(prompt, max_tokens=300)
    result = _parse_turn_intent_response(response, current_turn_text)
    return result


def _parse_turn_intent_response(response: str, turn_text: str) -> TurnIntentDecision:
    """Parse LLM response into TurnIntentDecision."""
    lines = response.strip().split('\n')

    # Defaults
    intent = "NON_QA"
    supports_qa = False
    confidence = 0.5
    is_follow_up = False
    follow_up_reason = None
    evidence_text = ""

    valid_intents = [
        "QUESTION_START", "QUESTION_CONTINUATION",
        "ANSWER_START", "ANSWER_CONTINUATION",
        "MODERATOR_TRANSITION", "NON_QA"
    ]

    for line in lines:
        upper_line = line.upper()
        if 'INTENT:' in upper_line:
            val = line.split(':', 1)[-1].strip().upper()
            # Handle variations
            val = val.replace("_", "_").strip()
            if val in valid_intents:
                intent = val
            elif "QUESTION" in val and "START" in val:
                intent = "QUESTION_START"
            elif "QUESTION" in val and "CONT" in val:
                intent = "QUESTION_CONTINUATION"
            elif "ANSWER" in val and "START" in val:
                intent = "ANSWER_START"
            elif "ANSWER" in val and "CONT" in val:
                intent = "ANSWER_CONTINUATION"
            elif "MODERATOR" in val or "TRANSITION" in val:
                intent = "MODERATOR_TRANSITION"
        elif 'SUPPORTS_QA:' in upper_line:
            supports_qa = 'YES' in upper_line
        elif 'CONFIDENCE:' in upper_line:
            try:
                confidence = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass
        elif 'IS_FOLLOW_UP:' in upper_line:
            is_follow_up = 'YES' in upper_line
        elif 'FOLLOW_UP_REASON:' in upper_line:
            val = line.split(':', 1)[-1].strip()
            if val.upper() not in ('NONE', 'N/A', ''):
                follow_up_reason = val[:200]
        elif 'EVIDENCE:' in upper_line:
            evidence_text = line.split(':', 1)[-1].strip()[:200]

    # Build evidence spans
    evidence_spans = []
    if evidence_text:
        evidence_spans.append(EvidenceSpan(
            text=evidence_text,
            source="turn_text",
            relevance="Key phrase for intent classification"
        ))

    # Determine supports_qa from intent if not explicitly set
    if not supports_qa and intent in ("QUESTION_START", "QUESTION_CONTINUATION",
                                       "ANSWER_START", "ANSWER_CONTINUATION"):
        supports_qa = True

    return TurnIntentDecision(
        intent=intent,
        supports_qa=supports_qa,
        confidence=confidence,
        evidence_spans=evidence_spans,
        is_follow_up=is_follow_up,
        follow_up_reason=follow_up_reason
    )


# =============================================================================
# Q&A Extraction Helpers (Legacy - kept for compatibility)
# =============================================================================

def classify_turn(
    speaker_name: str,
    turn_text: str,
    speaker_role: str,
    previous_turn_type: str = ""
) -> TurnClassification:
    """Classify a speaker turn as question, answer, or other.

    Args:
        speaker_name: Who is speaking
        turn_text: What they said
        speaker_role: Known role (analyst, management, etc.)
        previous_turn_type: Type of previous turn for context

    Returns:
        TurnClassification
    """
    # First apply deterministic rules
    has_question_mark = "?" in turn_text

    # Strong signals
    if speaker_role == "analyst" and has_question_mark:
        return TurnClassification(turn_type="question", confidence=0.95)

    if speaker_role == "management" and previous_turn_type == "question":
        return TurnClassification(turn_type="answer", confidence=0.9)

    if speaker_role == "moderator":
        return TurnClassification(turn_type="transition", confidence=0.85)

    # Use LLM for ambiguous cases
    prompt = f"""Classify this speaker turn from an earnings call:

Speaker: {speaker_name} (role: {speaker_role})
Previous turn type: {previous_turn_type or "START"}
Text: "{turn_text[:300]}"

Is this a:
QUESTION - asking for information
ANSWER - responding to a question
TRANSITION - moderator moving to next topic
OTHER - something else

Respond with ONE word: QUESTION, ANSWER, TRANSITION, or OTHER"""

    response = _invoke_llm(prompt, max_tokens=50)

    valid_types = ["question", "answer", "transition", "other"]
    turn_type, confidence = _parse_choice_response(response, valid_types)

    return TurnClassification(turn_type=turn_type, confidence=confidence)


def is_follow_up_question(
    current_question: str,
    previous_question: str,
    same_speaker: bool
) -> BinaryDecision:
    """Determine if current question is a follow-up to previous.

    Args:
        current_question: The current question text
        previous_question: The previous question text
        same_speaker: Whether same person is asking

    Returns:
        BinaryDecision
    """
    if not same_speaker:
        return BinaryDecision(decision="NO", confidence=0.95, reasoning="Different speakers")

    prompt = f"""Is this a FOLLOW-UP question?

Previous question: "{previous_question[:200]}"
Current question: "{current_question[:200]}"
Same speaker: Yes

A follow-up question:
- Continues the same topic
- Says "just to follow up" or "one more question"
- Asks for clarification on the previous answer

Answer YES or NO:"""

    response = _invoke_llm(prompt, max_tokens=50)
    is_yes, confidence, reasoning = _parse_binary_response(response)

    return BinaryDecision(
        decision="YES" if is_yes else "NO",
        confidence=confidence,
        reasoning=reasoning[:200]
    )


# =============================================================================
# Strategic Statement Helpers
# =============================================================================

def classify_statement_type(
    statement_text: str,
    speaker_name: str,
    section_type: str
) -> tuple[str, bool, float]:
    """Classify a strategic statement.

    Args:
        statement_text: The statement text
        speaker_name: Who said it
        section_type: Which section (opening_remarks, closing_remarks)

    Returns:
        Tuple of (statement_type, is_forward_looking, confidence)
    """
    prompt = f"""Classify this statement from an earnings call {section_type}:

Speaker: {speaker_name}
Statement: "{statement_text[:400]}"

Statement types:
- guidance: Forward-looking financial targets
- outlook: General business outlook
- strategic_initiative: New projects/plans
- operational_update: Current operations status
- financial_highlight: Past financial results
- risk_disclosure: Risks and challenges
- other: None of the above

Is it FORWARD-LOOKING (about future, not past)?

Respond in format:
TYPE: [guidance/outlook/strategic_initiative/operational_update/financial_highlight/risk_disclosure/other]
FORWARD_LOOKING: [YES/NO]
CONFIDENCE: [0.0 to 1.0]"""

    response = _invoke_llm(prompt)

    # Parse response
    stmt_type = "other"
    is_forward = False
    confidence = 0.5

    valid_types = ["guidance", "outlook", "strategic_initiative", "operational_update",
                   "financial_highlight", "risk_disclosure", "other"]

    lines = response.strip().split('\n')
    for line in lines:
        if 'TYPE:' in line.upper():
            type_str = line.split(':', 1)[-1].strip().lower()
            if type_str in valid_types:
                stmt_type = type_str
        elif 'FORWARD_LOOKING:' in line.upper():
            is_forward = 'YES' in line.upper()
        elif 'CONFIDENCE:' in line.upper():
            try:
                confidence = float(line.split(':', 1)[-1].strip())
            except ValueError:
                pass

    return stmt_type, is_forward, confidence


# =============================================================================
# CONTEXT-AWARE LLM FUNCTIONS (Document/Section Level)
# =============================================================================
# These functions give LLM FULL AUTHORITY to restructure outputs.
# LLM is NOT limited to binary decisions - it can rewrite structure.

class VerifiedSpeaker(BaseModel):
    """A speaker verified by LLM with full context."""
    canonical_name: str = Field(max_length=50, description="Clean human name (FirstName LastName)")
    role: Literal["moderator", "management", "analyst", "unknown"]
    title: Optional[str] = Field(None, max_length=50, description="Job title if clearly stated")
    company: Optional[str] = Field(None, max_length=50, description="Company affiliation if stated")
    aliases: list[str] = Field(default_factory=list, description="Other names that refer to this person")
    justification: str = Field(max_length=300, description="Why this speaker was verified/classified this way")


class SpeakerRegistryDecision(BaseModel):
    """LLM decision for entire speaker registry (document-level).

    LLM sees ALL candidates and makes GLOBAL decisions about:
    - Which candidates are real people
    - How to merge aliases
    - What role each speaker has
    - What the final clean registry should be
    """
    verified_speakers: list[VerifiedSpeaker] = Field(
        description="Final clean list of verified speakers"
    )
    rejected_candidates: list[dict] = Field(
        default_factory=list,
        description="Candidates rejected with reasons"
    )
    merge_decisions: list[dict] = Field(
        default_factory=list,
        description="Which candidates were merged and why"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class QAUnit(BaseModel):
    """A single Q&A unit identified by LLM.

    LLM has FULL AUTHORITY to decide exact boundaries.
    Single justification field explains the decision.
    """
    questioner: str = Field(description="Name of the person asking")
    question_text: str = Field(description="Complete question text")
    responders: list[str] = Field(description="Names of people responding")
    response_text: str = Field(description="Complete response text")
    start_line: int = Field(description="Starting line number in section")
    end_line: int = Field(description="Ending line number in section")
    is_follow_up: bool = Field(default=False)
    follow_up_of: Optional[int] = Field(None, description="Index of the Q&A this follows up")
    justification: str = Field(max_length=300, description="Why boundaries were drawn here")


class QAExtractionDecision(BaseModel):
    """LLM decision for Q&A extraction (section-level).

    LLM sees ENTIRE Q&A session and decides:
    - Exact question/answer boundaries
    - Which turns belong to which Q&A
    - Follow-up relationships
    - Multi-speaker answers
    """
    qa_units: list[QAUnit] = Field(
        description="Complete list of Q&A units in this section"
    )
    moderator_lines: list[int] = Field(
        default_factory=list,
        description="Line numbers that are moderator transitions"
    )
    non_qa_lines: list[int] = Field(
        default_factory=list,
        description="Line numbers that are not part of Q&A"
    )
    confidence: float = Field(ge=0.0, le=1.0)


def verify_speaker_registry_with_context(
    speaker_candidates: list[dict],
    opening_remarks_text: str,
    qa_session_text: str,
    metadata_hints: Optional[dict] = None,
) -> SpeakerRegistryDecision:
    """Verify entire speaker registry with full document context.

    LLM has FULL AUTHORITY to:
    - Delete candidates that are not real people
    - Rename canonical names
    - Merge or split speakers
    - Assign roles based on behavior patterns

    Args:
        speaker_candidates: All proposed speakers from heuristics with their turns
        opening_remarks_text: Full opening remarks section
        qa_session_text: Full Q&A session text
        metadata_hints: Optional hints from document metadata

    Returns:
        SpeakerRegistryDecision with final verified registry
    """
    # Build candidate summary for prompt
    candidate_summary = ""
    for i, cand in enumerate(speaker_candidates[:20]):  # Limit to avoid token overflow
        name = cand.get("name", "Unknown")
        turns = cand.get("turns", [])
        turn_samples = "\n    ".join([t[:150] + "..." if len(t) > 150 else t for t in turns[:3]])
        candidate_summary += f"""
CANDIDATE {i+1}: "{name}"
  Sample turns:
    {turn_samples}
"""

    prompt = f"""You are building a speaker registry for an earnings call transcript.

=== CRITICAL RULE: WHAT MAKES A REAL PERSON ===

A REAL PERSON does NOT need to have spoken.

If a name appears in ANY of these contexts, they ARE a REAL PERSON:
- Management lists or opening remarks
- Participant introductions
- Moderator announcements ("Next question is from...")
- Company executive listings

SILENCE DOES NOT MEAN INVALID. Keep silent management members as is_real_person=true.

=== WHAT TO REJECT (STRICT - ONLY THESE) ===

REJECT ONLY if the entry is clearly NOT A HUMAN PERSON:
1. DOCUMENT ARTIFACTS: "Scrip Code", "CIN", "ISIN", "NSE Code", "BSE Code", "Page 1 of 16"
2. GENERIC LABELS: "MANAGEMENT", "ANALYSTS", "Participants", "Speakers", "Business"
3. JOB TITLES WITHOUT A NAME: "Compliance Officer", "Company Secretary" (title only, no person name)
4. ABSTRACT NOUNS: "Business", "Industry", "Finance", "Mumbai", city names
5. SENTENCE FRAGMENTS: "Conference Call. As a reminder...", "Before we begin..."

NEVER REJECT because:
- "No speaking turn detected" - INVALID rejection reason
- "Management title listed but no speaking turn" - INVALID rejection reason
- "Name appears but no speaking turn" - INVALID rejection reason

If the text RESEMBLES A HUMAN NAME (e.g., "Alexander Poempner", "Thomas Kehl"), DO NOT REJECT.

=== SPEAKER CANDIDATES FROM HEURISTICS ===
{candidate_summary}

=== OPENING REMARKS CONTEXT (first 1500 chars) ===
{opening_remarks_text[:1500]}

=== Q&A SESSION CONTEXT (first 2000 chars) ===
{qa_session_text[:2000]}

=== ROLE ASSIGNMENT RULES ===
Assign roles using CONTEXT, not just speaking turns.

MODERATOR: Manages call flow, introduces speakers (usually "Operator" or "Moderator")
MANAGEMENT: Listed in management section, has corporate title, OR answers questions
ANALYST: Introduced by moderator as analyst, OR asks questions
UNKNOWN: Only if role cannot be inferred - but person is still REAL

Silent management members still get role="management".
Look at opening remarks for management listings and titles.

=== ALIAS HANDLING RULES (VERY STRICT) ===
- Canonical name MUST NOT appear in its own alias list
- Aliases must be DISTINCT textual variants that actually appear in the document
- DO NOT invent aliases
- If no real variants exist, aliases must be an empty list []

VALID: canonical_name="Alexander Poempner", aliases=["Mr. Poempner", "Alex Poempner"]
INVALID: canonical_name="Alexander Poempner", aliases=["Alexander Poempner"] <-- WRONG!

=== OUTPUT FORMAT ===
Respond with a JSON object:
{{
  "verified_speakers": [
    {{
      "canonical_name": "FirstName LastName",
      "role": "management|analyst|moderator|unknown",
      "title": "CEO" or null,
      "company": "Company Name" or null,
      "aliases": [],
      "justification": "Why this person was verified"
    }}
  ],
  "rejected_candidates": [
    {{"name": "rejected name", "reason": "Document artifact|Generic label|Job title without name|Abstract noun|Sentence fragment"}}
  ],
  "merge_decisions": [
    {{"merged": ["Name1", "Name2"], "into": "Canonical Name", "reason": "why merged"}}
  ],
  "confidence": 0.85
}}

DECISION OUTPUT REQUIREMENTS:
- For every candidate, return EITHER in verified_speakers OR rejected_candidates
- NEVER silently discard candidates
- Rejections MUST include an explicit non-person reason
- VALID rejection reasons: "Document artifact", "Generic label", "Job title without person name", "Abstract noun", "Sentence fragment"
- INVALID rejection reasons: "No speaking turn detected", "Management title but no speaking turn"

IMPORTANT: Output ONLY valid JSON. No explanations before or after."""

    response = _invoke_llm(prompt, max_tokens=2000)
    return _parse_speaker_registry_decision(response, speaker_candidates)


def _parse_speaker_registry_decision(
    response: str,
    original_candidates: list[dict]
) -> SpeakerRegistryDecision:
    """Parse LLM response into SpeakerRegistryDecision."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())

            verified = []
            for sp in data.get("verified_speakers", []):
                canonical = sp.get("canonical_name", "Unknown")[:50]
                raw_aliases = sp.get("aliases", [])

                # HARD RULE: Filter aliases - canonical name must NOT be in its own alias list
                # Also filter empty strings and duplicates
                clean_aliases = []
                canonical_lower = canonical.lower().strip()
                for alias in raw_aliases:
                    if not alias or not isinstance(alias, str):
                        continue
                    alias_clean = alias.strip()
                    if not alias_clean:
                        continue
                    # Skip if alias matches canonical name (case-insensitive)
                    if alias_clean.lower() == canonical_lower:
                        continue
                    # Skip duplicates
                    if alias_clean.lower() in [a.lower() for a in clean_aliases]:
                        continue
                    clean_aliases.append(alias_clean)

                verified.append(VerifiedSpeaker(
                    canonical_name=canonical,
                    role=sp.get("role", "unknown"),
                    title=sp.get("title"),
                    company=sp.get("company"),
                    aliases=clean_aliases,
                    justification=sp.get("justification", "")[:300],
                ))

            return SpeakerRegistryDecision(
                verified_speakers=verified,
                rejected_candidates=data.get("rejected_candidates", []),
                merge_decisions=data.get("merge_decisions", []),
                confidence=data.get("confidence", 0.7),
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("speaker_registry_json_parse_failed", error=str(e))

    # Fallback: return original candidates as-is
    verified = []
    for cand in original_candidates[:15]:
        name = cand.get("name", "Unknown")
        if len(name) < 50 and not any(w in name.lower() for w in ['conference', 'call', 'reminder']):
            verified.append(VerifiedSpeaker(
                canonical_name=name,
                role="unknown",
                title=None,
                company=None,
                aliases=[],
                justification="LLM parse failed - using heuristic candidate",
            ))

    return SpeakerRegistryDecision(
        verified_speakers=verified,
        rejected_candidates=[],
        merge_decisions=[],
        confidence=0.3,
    )


def extract_qa_units_from_section(
    section_text: str,
    speaker_turns: list[dict],
    verified_registry: dict[str, str],
    section_start_page: int,
) -> QAExtractionDecision:
    """Extract Q&A units from entire section with full context.

    LLM sees ENTIRE Q&A session and has FULL AUTHORITY to:
    - Decide exact question/answer boundaries
    - Group multi-part questions
    - Handle follow-ups naturally
    - Support multi-speaker answers

    Args:
        section_text: Full Q&A section text
        speaker_turns: List of {speaker, text, line_num} dicts
        verified_registry: Map of speaker names to roles
        section_start_page: Starting page number

    Returns:
        QAExtractionDecision with all Q&A units
    """
    # Build turn summary with line numbers
    turn_summary = ""
    for i, turn in enumerate(speaker_turns[:50]):  # Limit to avoid overflow
        speaker = turn.get("speaker", "Unknown")
        role = verified_registry.get(speaker, "unknown")
        text = turn.get("text", "")[:200]
        line = turn.get("line_num", i)
        turn_summary += f"[Line {line}] {speaker} ({role}): {text}\n"

    prompt = f"""You are extracting Q&A units from an earnings call transcript.

TASK: Identify ALL question-answer pairs in this Q&A session.

You have FULL AUTHORITY to:
- Decide EXACTLY where each question starts and ends
- Decide EXACTLY where each answer starts and ends
- Group multi-part questions from the same analyst
- Link follow-up questions
- Handle multiple responders per question

=== SPEAKER TURNS (with line numbers) ===
{turn_summary}

=== EXTRACTION RULES ===
1. A QUESTION is asked by an analyst (role=analyst)
2. An ANSWER is given by management (role=management)
3. MODERATOR speech is NOT part of Q&A - it's transition
4. Questions may span multiple turns if same speaker continues
5. Answers may include multiple management speakers
6. NEVER miss a Q&A - if unsure, over-split rather than under-split
7. Link follow-ups when same analyst asks again after getting an answer

=== OUTPUT FORMAT ===
Respond with a JSON object:
{{
  "qa_units": [
    {{
      "questioner": "Analyst Name",
      "question_text": "Complete question text...",
      "responders": ["Executive Name 1", "Executive Name 2"],
      "response_text": "Complete response text...",
      "start_line": 5,
      "end_line": 12,
      "is_follow_up": false,
      "follow_up_of": null,
      "justification": "Why these boundaries were chosen"
    }}
  ],
  "moderator_lines": [1, 15, 30],
  "non_qa_lines": [2],
  "confidence": 0.9
}}

IMPORTANT:
- Output ONLY valid JSON
- Include ALL Q&A units found
- question_text and response_text should be COMPLETE, not truncated
- For follow-ups, follow_up_of is the INDEX (0-based) of the Q&A it follows"""

    response = _invoke_llm(prompt, max_tokens=3000)
    return _parse_qa_extraction_decision(response, speaker_turns)


def _parse_qa_extraction_decision(
    response: str,
    speaker_turns: list[dict]
) -> QAExtractionDecision:
    """Parse LLM response into QAExtractionDecision."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())

            qa_units = []
            for qa in data.get("qa_units", []):
                qa_units.append(QAUnit(
                    questioner=qa.get("questioner", "Unknown"),
                    question_text=qa.get("question_text", ""),
                    responders=qa.get("responders", ["Management"]),
                    response_text=qa.get("response_text", ""),
                    start_line=qa.get("start_line", 0),
                    end_line=qa.get("end_line", 0),
                    is_follow_up=qa.get("is_follow_up", False),
                    follow_up_of=qa.get("follow_up_of"),
                    justification=qa.get("justification", "")[:300],
                ))

            return QAExtractionDecision(
                qa_units=qa_units,
                moderator_lines=data.get("moderator_lines", []),
                non_qa_lines=data.get("non_qa_lines", []),
                confidence=data.get("confidence", 0.7),
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("qa_extraction_json_parse_failed", error=str(e))

    # Fallback: return empty result
    return QAExtractionDecision(
        qa_units=[],
        moderator_lines=[],
        non_qa_lines=[],
        confidence=0.3,
    )


