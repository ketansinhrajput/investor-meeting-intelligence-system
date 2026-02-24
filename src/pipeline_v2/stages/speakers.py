"""Stage 3: Speaker Registry - Document-Level Speaker Extraction with LLM Authority.

CONTEXT-AWARE HYBRID INTELLIGENCE APPROACH:
- Phase A: Deterministic candidate generation with HIGH RECALL
- Phase B: DOCUMENT-LEVEL LLM verification (LLM sees ALL candidates at once)

Rules propose -> LLM restructures -> Code enforces invariants

PHILOSOPHY:
- High recall first, precision second
- LLM has FULL AUTHORITY to delete, rename, merge, split speakers
- Role assignment based on BEHAVIOR (who asks vs who answers)
- Every LLM decision must emit justification trace

HARD RULES (enforced by code AFTER LLM decision):
- The literal speaker "Moderator" is a SINGLE canonical speaker
- Moderator NEVER has a title
- Titles (e.g., CFO) require explicit textual evidence
- One CFO per company per call unless explicitly stated otherwise
- Sentence-like names (>5 words or verb phrases) must be rejected

LLM has FULL AUTHORITY to:
- Delete candidates that are not real people
- Rename canonical names
- Merge or split speakers
- Assign roles based on behavior patterns
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog
from rapidfuzz import fuzz

from src.pipeline_v2.llm_helpers import (
    verify_speaker_registry_with_context,
    SpeakerRegistryDecision,
    VerifiedSpeaker,
    EvidenceSpan,
)
from src.pipeline_v2.models import (
    BoundaryDetectionResult,
    ExtractedMetadata,
    SectionType,
    SpeakerInfo,
    SpeakerRegistry,
    SpeakerRole,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# ENHANCED Trace Dataclasses (with evidence_spans)
# =============================================================================

@dataclass
class NameCandidate:
    """A candidate speaker name found in the transcript."""
    raw_name: str
    normalized_name: str
    is_valid: bool
    rejection_reason: Optional[str] = None
    section_id: str = ""
    context_snippet: str = ""
    # Evidence spans supporting validity/rejection
    evidence_spans: list[dict] = field(default_factory=list)


@dataclass
class SpeakerVerificationRecord:
    """Record of LLM verification for a speaker candidate."""
    candidate_name: str
    is_real_person: bool
    canonical_name: str
    role: str
    role_confidence: float
    title: Optional[str]
    title_verified: bool
    company: Optional[str]
    merge_with: Optional[str]
    merge_confidence: float
    evidence_spans: list[dict] = field(default_factory=list)
    reasoning: str = ""
    rejection_reason: Optional[str] = None
    verified_by_llm: bool = False


@dataclass
class AliasMergeDecision:
    """Record of an alias merge decision."""
    name1: str
    name2: str
    merged: bool
    reason: str
    used_llm: bool
    llm_confidence: Optional[float] = None
    evidence_spans: list[dict] = field(default_factory=list)


@dataclass
class TitleAssignmentRecord:
    """Record of title assignment with evidence."""
    speaker_name: str
    title: Optional[str]
    title_verified: bool
    evidence_text: str = ""
    rejection_reason: Optional[str] = None


@dataclass
class SpeakerRegistryTrace:
    """Complete trace of speaker registry building for inspection.

    This trace allows humans to audit:
    - Which names were found and which were rejected
    - LLM verification decisions for each speaker
    - How roles and titles were assigned
    - Which aliases were merged and why
    """
    # Phase A: Candidate generation
    candidates_generated: list[NameCandidate] = field(default_factory=list)
    candidates_passed_phase_a: int = 0
    candidates_rejected_phase_a: int = 0

    # Phase B: LLM verification
    verification_decisions: list[SpeakerVerificationRecord] = field(default_factory=list)
    speakers_verified_by_llm: int = 0
    speakers_rejected_by_llm: int = 0

    # Alias merge decisions
    alias_merge_decisions: list[AliasMergeDecision] = field(default_factory=list)

    # Title assignments
    title_assignments: list[TitleAssignmentRecord] = field(default_factory=list)

    # Hard rule enforcement
    hard_rule_enforcements: list[dict] = field(default_factory=list)

    # LLM usage stats
    llm_calls_made: int = 0
    total_candidates: int = 0
    final_speakers: int = 0


# =============================================================================
# HARD RULES (enforced by code, not LLM)
# =============================================================================

# Words that indicate content, not a speaker name
CONTENT_INDICATOR_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'that', 'which',
    'who', 'whom', 'this', 'these', 'those', 'what', 'conference', 'call',
    'reminder', 'hand', 'over', 'begin', 'management', 'thank', 'thanks',
    'you', 'we', 'i', 'they', 'he', 'she', 'now', 'before', 'after',
    'good', 'morning', 'afternoon', 'evening', 'ladies', 'gentlemen',
}

# Verbs that indicate sentence fragments
VERB_INDICATORS = {
    'said', 'says', 'noted', 'mentioned', 'stated', 'explained',
    'continued', 'added', 'concluded', 'announced', 'reported',
    'believes', 'thinks', 'expects', 'anticipates', 'projects',
}

# Executive titles that can only appear once per call
UNIQUE_TITLES = {'cfo', 'ceo', 'coo', 'cto', 'cmo', 'chairman', 'president'}


def _enforce_hard_rules_on_name(name: str) -> tuple[bool, Optional[str]]:
    """Apply HARD RULES to reject invalid speaker names.

    These rules are enforced by code and cannot be overridden by LLM.

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if not name:
        return False, "empty_name"

    if '\n' in name:
        return False, "contains_newline"

    name = name.strip()

    # HARD RULE: Length limits
    if len(name) < 2:
        return False, "too_short"
    if len(name) > 60:
        return False, "too_long"

    words = name.split()

    # HARD RULE: Max 5 words (>5 = sentence fragment)
    if len(words) > 5:
        return False, "sentence_fragment:too_many_words"

    # HARD RULE: First word check
    first_word = words[0].lower().rstrip('.,;:')
    if first_word in CONTENT_INDICATOR_WORDS:
        return False, f"starts_with_content_word:{first_word}"

    # HARD RULE: Verb phrases are not names
    for word in words:
        if word.lower().rstrip('.,;:') in VERB_INDICATORS:
            return False, f"contains_verb:{word}"

    # HARD RULE: Multiple content words = probably a sentence
    content_word_count = sum(
        1 for w in words
        if w.lower().rstrip('.,;:') in CONTENT_INDICATOR_WORDS
    )
    if content_word_count >= 2:
        return False, "multiple_content_words"

    # HARD RULE: No sentence-ending punctuation
    if name.rstrip().endswith(('.', '!', '?')):
        return False, "sentence_ending_punctuation"

    # HARD RULE: "Moderator" and "Operator" are always valid
    if name.lower() in ('moderator', 'operator'):
        return True, None

    return True, None


def _enforce_hard_rules_on_title(
    speaker_name: str,
    proposed_title: Optional[str],
    title_evidence_text: str,
    existing_title_holders: dict[str, str],
    trace: SpeakerRegistryTrace,
) -> tuple[Optional[str], bool]:
    """Apply HARD RULES to title assignment.

    HARD RULES:
    - Moderator NEVER has a title
    - Titles require explicit textual evidence
    - One CFO/CEO per company unless explicitly stated

    Returns:
        Tuple of (verified_title or None, is_verified)
    """
    # HARD RULE: Moderator never has title
    if speaker_name.lower() in ('moderator', 'operator'):
        if proposed_title:
            trace.hard_rule_enforcements.append({
                "rule": "moderator_no_title",
                "speaker": speaker_name,
                "proposed_title": proposed_title,
                "action": "rejected",
            })
        return None, False

    if not proposed_title:
        return None, False

    proposed_lower = proposed_title.lower()

    # HARD RULE: Title must appear in evidence text
    if proposed_lower not in title_evidence_text.lower():
        trace.hard_rule_enforcements.append({
            "rule": "title_requires_evidence",
            "speaker": speaker_name,
            "proposed_title": proposed_title,
            "action": "rejected_no_evidence",
        })
        trace.title_assignments.append(TitleAssignmentRecord(
            speaker_name=speaker_name,
            title=None,
            title_verified=False,
            rejection_reason="Title not found in evidence text",
        ))
        return None, False

    # HARD RULE: Unique titles (CEO, CFO) can only appear once
    for unique_title in UNIQUE_TITLES:
        if unique_title in proposed_lower:
            # Check if someone else already has this title
            existing_holder = existing_title_holders.get(unique_title)
            if existing_holder and existing_holder != speaker_name:
                trace.hard_rule_enforcements.append({
                    "rule": "unique_title_constraint",
                    "speaker": speaker_name,
                    "proposed_title": proposed_title,
                    "existing_holder": existing_holder,
                    "action": "rejected_duplicate",
                })
                trace.title_assignments.append(TitleAssignmentRecord(
                    speaker_name=speaker_name,
                    title=None,
                    title_verified=False,
                    rejection_reason=f"{unique_title.upper()} already assigned to {existing_holder}",
                ))
                return None, False
            else:
                # Record this speaker as the title holder
                existing_title_holders[unique_title] = speaker_name

    # Title is valid
    trace.title_assignments.append(TitleAssignmentRecord(
        speaker_name=speaker_name,
        title=proposed_title,
        title_verified=True,
        evidence_text=title_evidence_text[:200],
    ))
    return proposed_title, True


# =============================================================================
# Phase A: Deterministic Candidate Generation
# =============================================================================

def _normalize_name(name: str) -> str:
    """Normalize speaker name for comparison."""
    # Remove titles and honorifics
    name = re.sub(r"(?i)^(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s*", "", name)
    # Remove extra whitespace and newlines
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _is_moderator_name(name: str) -> bool:
    """Check if name refers to moderator/operator."""
    name_lower = name.lower().strip()
    return name_lower in ('moderator', 'operator', 'conference operator', 'call operator')


def _collect_speaker_context(
    candidate_name: str,
    occurrences: list[dict],
    all_sections: list,
    full_text: str,
) -> dict:
    """Collect comprehensive context for LLM verification.

    Returns dict with:
    - all_turns: List of text spoken by this speaker
    - moderator_introductions: Lines where moderator introduced this speaker
    - opening_remarks_context: Text from opening remarks
    - similar_names: Names that might be aliases
    """
    context = {
        "all_turns": [],
        "moderator_introductions": [],
        "opening_remarks_context": "",
        "similar_names": [],
    }

    # Collect all turns by this speaker
    for occ in occurrences:
        turn_text = occ.get("context", "")[:500]
        if turn_text:
            context["all_turns"].append(turn_text)

    # Look for moderator introductions
    name_escaped = re.escape(candidate_name)
    intro_patterns = [
        rf"(?i)(?:next\s+)?question\s+(?:is\s+)?(?:from|comes\s+from)\s+(?:the\s+line\s+of\s+)?[^.]*{name_escaped}",
        rf"(?i)(?:we\s+have|let's\s+go\s+to)\s+{name_escaped}",
        rf"(?i){name_escaped}[,\s]+(?:from|of|with)\s+[A-Z][a-zA-Z\s]+",
    ]

    for pattern in intro_patterns:
        for match in re.finditer(pattern, full_text):
            start = max(0, match.start() - 50)
            end = min(len(full_text), match.end() + 100)
            context["moderator_introductions"].append(full_text[start:end])

    # Get opening remarks context
    for section in all_sections:
        if section.section_type == SectionType.OPENING_REMARKS:
            context["opening_remarks_context"] = section.raw_text[:1000]
            break

    return context


def _check_name_similarity(name1: str, name2: str) -> tuple[bool, float]:
    """Check if two names might refer to the same person.

    Returns:
        Tuple of (is_similar, similarity_score)
    """
    norm1 = _normalize_name(name1).lower()
    norm2 = _normalize_name(name2).lower()

    # Exact match
    if norm1 == norm2:
        return True, 100.0

    # One contains the other
    if norm1 in norm2 or norm2 in norm1:
        return True, 95.0

    # Same first and last name
    parts1 = norm1.split()
    parts2 = norm2.split()
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[0] == parts2[0] and parts1[-1] == parts2[-1]:
            return True, 90.0

    # Fuzzy match
    score = fuzz.ratio(norm1, norm2)
    return score >= 85, float(score)


# =============================================================================
# Main Registry Building (Document-Level Hybrid)
# =============================================================================

def build_speaker_registry(
    boundary_result: BoundaryDetectionResult,
    metadata: Optional[ExtractedMetadata] = None,
    full_text: Optional[str] = None,
    use_llm: bool = True,
) -> tuple[SpeakerRegistry, SpeakerRegistryTrace]:
    """Build speaker registry using two-phase extraction.

    PHASE A: Deterministic candidate generation
    - Regex-based name extraction
    - Hard rule validation (>5 words, verb phrases, etc.)
    - Basic deduplication

    PHASE B: LLM-based contextual verification
    - Full document context provided to LLM
    - LLM can reject, correct roles, verify titles, prevent merges
    - Hard rules enforced by code after LLM decision

    Args:
        boundary_result: Result from boundary detection stage.
        metadata: Optional metadata with participant hints.
        full_text: Optional full text for additional context.
        use_llm: Whether to use LLM for Phase B verification.

    Returns:
        Tuple of (SpeakerRegistry, SpeakerRegistryTrace) for inspectability.
    """
    logger.info("speaker_registry_start", section_count=len(boundary_result.sections))

    trace = SpeakerRegistryTrace()
    speakers: dict[str, SpeakerInfo] = {}
    speaker_counter = 0
    title_holders: dict[str, str] = {}  # Track unique title assignments

    # =========================================================================
    # PHASE A: Deterministic Candidate Generation
    # =========================================================================

    logger.info("phase_a_candidate_generation_start")

    # Collect all speaker occurrences with context
    speaker_occurrences: dict[str, list[dict]] = defaultdict(list)

    for section in boundary_result.sections:
        for speaker_name in section.detected_speakers:
            # Apply HARD RULES
            is_valid, rejection_reason = _enforce_hard_rules_on_name(speaker_name)

            candidate = NameCandidate(
                raw_name=speaker_name,
                normalized_name=_normalize_name(speaker_name),
                is_valid=is_valid,
                rejection_reason=rejection_reason,
                section_id=section.section_id,
                context_snippet=section.raw_text[:200] if section.raw_text else "",
            )
            trace.candidates_generated.append(candidate)
            trace.total_candidates += 1

            if not is_valid:
                trace.candidates_rejected_phase_a += 1
                logger.debug("phase_a_rejected", name=speaker_name[:50], reason=rejection_reason)
                continue

            trace.candidates_passed_phase_a += 1
            speaker_occurrences[speaker_name].append({
                "section_id": section.section_id,
                "section_type": section.section_type,
                "page": section.start_page,
                "context": section.raw_text[:500] if section.raw_text else "",
            })

    logger.info(
        "phase_a_complete",
        candidates=trace.total_candidates,
        passed=trace.candidates_passed_phase_a,
        rejected=trace.candidates_rejected_phase_a,
    )

    # =========================================================================
    # HARD RULE: Single Moderator Entry
    # =========================================================================

    moderator_occurrences = []
    non_moderator_occurrences: dict[str, list[dict]] = {}

    for name, occs in speaker_occurrences.items():
        if _is_moderator_name(name):
            moderator_occurrences.extend(occs)
        else:
            non_moderator_occurrences[name] = occs

    if moderator_occurrences:
        moderator_id = f"speaker_{speaker_counter:03d}"
        speakers[moderator_id] = SpeakerInfo(
            speaker_id=moderator_id,
            canonical_name="Moderator",
            aliases=[],
            role=SpeakerRole.MODERATOR,
            title=None,  # HARD RULE: Moderator never has title
            company=None,
            turn_count=len(moderator_occurrences),
            first_appearance_page=min(occ["page"] for occ in moderator_occurrences),
        )
        speaker_counter += 1

        trace.hard_rule_enforcements.append({
            "rule": "single_moderator",
            "action": "consolidated",
            "occurrences": len(moderator_occurrences),
        })
        trace.verification_decisions.append(SpeakerVerificationRecord(
            candidate_name="Moderator",
            is_real_person=True,
            canonical_name="Moderator",
            role="moderator",
            role_confidence=1.0,
            title=None,
            title_verified=False,
            company=None,
            merge_with=None,
            merge_confidence=0.0,
            reasoning="HARD RULE: Single moderator entry enforced by code",
            verified_by_llm=False,
        ))

    # =========================================================================
    # PHASE B: DOCUMENT-LEVEL LLM Verification
    # =========================================================================
    # LLM sees ALL candidates at once and makes GLOBAL decisions

    if use_llm and non_moderator_occurrences:
        logger.info("phase_b_document_level_verification", candidates=len(non_moderator_occurrences))

        # Collect ALL candidates with their turns for document-level LLM
        all_candidates: list[dict] = []
        for name, occurrences in non_moderator_occurrences.items():
            turns = []
            for occ in occurrences[:5]:  # Max 5 turns per speaker
                context = occ.get("context", "")
                if context:
                    turns.append(context[:300])
            all_candidates.append({
                "name": name,
                "turns": turns,
                "occurrences": len(occurrences),
                "first_page": min(occ["page"] for occ in occurrences),
            })

        # Collect section texts for context
        opening_remarks = ""
        qa_session = ""
        for section in boundary_result.sections:
            if section.section_type == SectionType.OPENING_REMARKS:
                opening_remarks = section.raw_text[:2000] if section.raw_text else ""
            elif section.section_type == SectionType.QA_SESSION:
                qa_session = section.raw_text[:3000] if section.raw_text else ""

        # SINGLE LLM CALL: Document-level verification
        trace.llm_calls_made += 1
        try:
            llm_decision = verify_speaker_registry_with_context(
                speaker_candidates=all_candidates,
                opening_remarks_text=opening_remarks,
                qa_session_text=qa_session,
                metadata_hints=None,
            )

            # Process LLM decision
            trace.speakers_verified_by_llm = len(llm_decision.verified_speakers)
            trace.speakers_rejected_by_llm = len(llm_decision.rejected_candidates)

            # Record rejected candidates in trace
            for rejected in llm_decision.rejected_candidates:
                trace.verification_decisions.append(SpeakerVerificationRecord(
                    candidate_name=rejected.get("name", "Unknown"),
                    is_real_person=False,
                    canonical_name="REJECTED",
                    role="unknown",
                    role_confidence=0.0,
                    title=None,
                    title_verified=False,
                    company=None,
                    merge_with=None,
                    merge_confidence=0.0,
                    reasoning=rejected.get("reason", "Rejected by document-level LLM"),
                    rejection_reason=rejected.get("reason"),
                    verified_by_llm=True,
                ))

            # Record merge decisions in trace
            for merge in llm_decision.merge_decisions:
                merged_names = merge.get("merged", [])
                if len(merged_names) >= 2:
                    trace.alias_merge_decisions.append(AliasMergeDecision(
                        name1=merged_names[0],
                        name2=merged_names[1] if len(merged_names) > 1 else "",
                        merged=True,
                        reason=merge.get("reason", "LLM document-level merge"),
                        used_llm=True,
                        llm_confidence=llm_decision.confidence,
                    ))

            # Build speakers from LLM-verified list
            for verified in llm_decision.verified_speakers:
                # Record verification in trace
                trace.verification_decisions.append(SpeakerVerificationRecord(
                    candidate_name=verified.canonical_name,
                    is_real_person=True,
                    canonical_name=verified.canonical_name,
                    role=verified.role,
                    role_confidence=llm_decision.confidence,
                    title=verified.title,
                    title_verified=verified.title is not None,
                    company=verified.company,
                    merge_with=None,
                    merge_confidence=0.0,
                    reasoning=verified.justification,
                    verified_by_llm=True,
                    evidence_spans=[{
                        "text": verified.justification[:200],
                        "source": "llm_document_level",
                        "relevance": "LLM justification for speaker verification"
                    }],
                ))

                # Apply HARD RULES after LLM decision
                # Find original occurrences for this speaker (check aliases too)
                speaker_occs = []
                for alias in [verified.canonical_name] + verified.aliases:
                    if alias in non_moderator_occurrences:
                        speaker_occs.extend(non_moderator_occurrences[alias])

                if not speaker_occs:
                    # Fallback: try fuzzy match
                    for cand_name, occs in non_moderator_occurrences.items():
                        if (cand_name.lower() in verified.canonical_name.lower() or
                            verified.canonical_name.lower() in cand_name.lower()):
                            speaker_occs.extend(occs)
                            break

                # Collect context for title verification
                all_context = " ".join([
                    occ.get("context", "")[:200] for occ in speaker_occs[:5]
                ])

                # HARD RULE: Validate title with evidence
                verified_title, title_is_verified = _enforce_hard_rules_on_title(
                    speaker_name=verified.canonical_name,
                    proposed_title=verified.title,
                    title_evidence_text=all_context,
                    existing_title_holders=title_holders,
                    trace=trace,
                )

                # Create speaker entry
                role_enum = {
                    "moderator": SpeakerRole.MODERATOR,
                    "management": SpeakerRole.MANAGEMENT,
                    "analyst": SpeakerRole.ANALYST,
                    "unknown": SpeakerRole.UNKNOWN,
                }.get(verified.role, SpeakerRole.UNKNOWN)

                # HARD RULE: Clean aliases - canonical name must NOT be in its own alias list
                clean_aliases = [
                    a for a in verified.aliases
                    if a and a.strip()
                    and a.strip().lower() != verified.canonical_name.lower()
                ]

                speaker_id = f"speaker_{speaker_counter:03d}"
                speakers[speaker_id] = SpeakerInfo(
                    speaker_id=speaker_id,
                    canonical_name=verified.canonical_name,
                    aliases=clean_aliases,
                    role=role_enum,
                    title=verified_title,  # Only hard-rule-verified titles
                    company=verified.company,
                    turn_count=len(speaker_occs) if speaker_occs else 1,
                    first_appearance_page=min((occ["page"] for occ in speaker_occs), default=1),
                )
                speaker_counter += 1

        except Exception as e:
            logger.warning("document_level_verification_failed", error=str(e))
            # Fallback: use heuristic-only speakers
            for name, occurrences in non_moderator_occurrences.items():
                normalized = _normalize_name(name)
                trace.verification_decisions.append(SpeakerVerificationRecord(
                    candidate_name=name,
                    is_real_person=True,
                    canonical_name=normalized,
                    role="unknown",
                    role_confidence=0.3,
                    title=None,
                    title_verified=False,
                    company=None,
                    merge_with=None,
                    merge_confidence=0.0,
                    reasoning=f"LLM failed, using heuristic: {str(e)[:100]}",
                    verified_by_llm=False,
                ))

                # HARD RULE: Only add alias if different from canonical name
                fallback_aliases = []
                if name != normalized and name.lower() != normalized.lower():
                    fallback_aliases = [name]

                speaker_id = f"speaker_{speaker_counter:03d}"
                speakers[speaker_id] = SpeakerInfo(
                    speaker_id=speaker_id,
                    canonical_name=normalized,
                    aliases=fallback_aliases,
                    role=SpeakerRole.UNKNOWN,
                    title=None,
                    company=None,
                    turn_count=len(occurrences),
                    first_appearance_page=min(occ["page"] for occ in occurrences),
                )
                speaker_counter += 1

    elif not use_llm:
        # LLM disabled: use deterministic processing only
        logger.info("phase_b_skipped_llm_disabled")
        for name, occurrences in non_moderator_occurrences.items():
            normalized = _normalize_name(name)

            trace.verification_decisions.append(SpeakerVerificationRecord(
                candidate_name=name,
                is_real_person=True,
                canonical_name=normalized,
                role="unknown",
                role_confidence=0.5,
                title=None,
                title_verified=False,
                company=None,
                merge_with=None,
                merge_confidence=0.0,
                reasoning="LLM disabled - using deterministic values",
                verified_by_llm=False,
            ))

            # HARD RULE: Only add alias if different from canonical name
            no_llm_aliases = []
            if name != normalized and name.lower() != normalized.lower():
                no_llm_aliases = [name]

            speaker_id = f"speaker_{speaker_counter:03d}"
            speakers[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                canonical_name=normalized,
                aliases=no_llm_aliases,
                role=SpeakerRole.UNKNOWN,
                title=None,
                company=None,
                turn_count=len(occurrences),
                first_appearance_page=min(occ["page"] for occ in occurrences),
            )
            speaker_counter += 1

    # =========================================================================
    # Build Final Registry
    # =========================================================================

    management_count = sum(1 for s in speakers.values() if s.role == SpeakerRole.MANAGEMENT)
    analyst_count = sum(1 for s in speakers.values() if s.role == SpeakerRole.ANALYST)
    moderator_count = sum(1 for s in speakers.values() if s.role == SpeakerRole.MODERATOR)

    trace.final_speakers = len(speakers)

    registry = SpeakerRegistry(
        speakers=speakers,
        total_speakers=len(speakers),
        management_count=management_count,
        analyst_count=analyst_count,
    )

    logger.info(
        "speaker_registry_complete",
        total_speakers=len(speakers),
        management=management_count,
        analysts=analyst_count,
        moderators=moderator_count,
        llm_calls=trace.llm_calls_made,
        verified_by_llm=trace.speakers_verified_by_llm,
        rejected_by_llm=trace.speakers_rejected_by_llm,
        hard_rules_enforced=len(trace.hard_rule_enforcements),
    )

    return registry, trace
