"""Intermediate data models for pipeline v2.

These models define the contracts between pipeline stages.
Philosophy: "Recall and structure first, interpretation later."

Stage Flow:
1. Metadata Extraction    → ExtractedMetadata
2. Boundary Detection     → List[TranscriptSection]
3. Speaker Registry       → SpeakerRegistry
4. Q&A Extraction         → List[ExtractedQAUnit]
5. Strategic Extraction   → List[ExtractedStrategicStatement]
6. Enrichment            → List[EnrichedQAUnit], List[EnrichedStrategicStatement]
7. Report Assembly       → Final StructuredReport
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums for v2 Pipeline
# =============================================================================

class SectionType(str, Enum):
    """Types of transcript sections detected by boundary detection."""

    OPENING_REMARKS = "opening_remarks"      # CEO/CFO prepared statements
    QA_SESSION = "qa_session"                # Q&A exchange block
    CLOSING_REMARKS = "closing_remarks"      # Final statements after Q&A
    TRANSITION = "transition"                # Moderator handoffs between sections
    UNKNOWN = "unknown"                      # Could not determine section type


class SpeakerRole(str, Enum):
    """Role classification for call participants."""

    MODERATOR = "moderator"                  # Conference operator, IR contact
    MANAGEMENT = "management"                # CEO, CFO, executives
    ANALYST = "analyst"                      # Investment analysts, investors
    UNKNOWN = "unknown"                      # Role could not be determined


class DetectionSignal(str, Enum):
    """Signals that triggered boundary detection."""

    MODERATOR_CUE = "moderator_cue"          # "Next question from..."
    SPEAKER_CHANGE = "speaker_change"        # New speaker started talking
    QUESTION_PATTERN = "question_pattern"    # Question mark, interrogative
    SECTION_HEADER = "section_header"        # Explicit section markers
    TOPIC_SHIFT = "topic_shift"              # Major topic change detected


# =============================================================================
# Stage 1: Metadata Extraction
# =============================================================================

class ExtractedMetadata(BaseModel):
    """Output of metadata extraction stage.

    Extracted from first 1-2 pages of transcript only.
    All fields nullable - extraction is best-effort.
    """

    company_name: Optional[str] = Field(
        None, description="Full company name as stated"
    )
    ticker_symbol: Optional[str] = Field(
        None, description="Stock ticker if mentioned"
    )
    fiscal_quarter: Optional[str] = Field(
        None, description="Q1, Q2, Q3, or Q4"
    )
    fiscal_year: Optional[int] = Field(
        None, description="Fiscal year (e.g., 2024)"
    )
    call_date: Optional[date] = Field(
        None, description="Date of the call"
    )
    call_title: Optional[str] = Field(
        None, description="Full title of the call"
    )
    mentioned_participants: list[str] = Field(
        default_factory=list,
        description="Names mentioned in header/intro (for speaker registry seeding)"
    )

    # Source tracking
    source_pages: tuple[int, int] = Field(
        default=(1, 2), description="Pages used for extraction"
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Model confidence in extraction"
    )


# =============================================================================
# Stage 2: Boundary Detection
# =============================================================================

class TranscriptSection(BaseModel):
    """A detected section/block of the transcript.

    Output of boundary detection stage.
    Goal: High recall - better to over-segment than miss content.
    """

    section_id: str = Field(
        description="Unique identifier (e.g., 'section_001')"
    )
    section_type: SectionType = Field(
        description="Type of section detected"
    )

    # Location in document
    start_page: int = Field(ge=1, description="Starting page number")
    end_page: int = Field(ge=1, description="Ending page number")
    char_offset_start: int = Field(ge=0, description="Start character offset in full text")
    char_offset_end: int = Field(ge=0, description="End character offset in full text")

    # Content
    raw_text: str = Field(description="Full text content of this section")

    # Detection metadata
    detected_speakers: list[str] = Field(
        default_factory=list,
        description="Speaker names detected in this section"
    )
    detection_signals: list[DetectionSignal] = Field(
        default_factory=list,
        description="What triggered this boundary detection"
    )
    detection_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Model confidence in boundary detection"
    )

    # Linking
    sequence_number: int = Field(
        ge=0, description="Order in transcript (0-indexed)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "section_id": "section_003",
                "section_type": "qa_session",
                "start_page": 5,
                "end_page": 6,
                "char_offset_start": 12500,
                "char_offset_end": 15200,
                "raw_text": "Moderator: Next question from John Smith...",
                "detected_speakers": ["Moderator", "John Smith", "CEO"],
                "detection_signals": ["moderator_cue", "speaker_change"],
                "detection_confidence": 0.95,
                "sequence_number": 3
            }
        }


class BoundaryDetectionResult(BaseModel):
    """Complete output of boundary detection stage."""

    sections: list[TranscriptSection] = Field(
        default_factory=list,
        description="All detected sections in order"
    )
    total_sections: int = Field(
        default=0, description="Total number of sections detected"
    )
    qa_section_count: int = Field(
        default=0, description="Number of Q&A sections detected"
    )
    opening_remarks_count: int = Field(
        default=0, description="Number of opening remarks sections"
    )

    # Validation helpers
    coverage_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percent of document covered by detected sections"
    )
    gaps_detected: int = Field(
        default=0, description="Number of gaps between sections (potential missed content)"
    )


# =============================================================================
# Stage 3: Speaker Registry
# =============================================================================

class SpeakerInfo(BaseModel):
    """Information about a single speaker."""

    speaker_id: str = Field(description="Unique identifier (e.g., 'speaker_001')")
    canonical_name: str = Field(description="Primary/canonical name")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other name variations seen (e.g., 'Mr. Patel', 'Tarak')"
    )
    role: SpeakerRole = Field(
        default=SpeakerRole.UNKNOWN,
        description="Inferred role"
    )
    title: Optional[str] = Field(
        None, description="Job title if detected (e.g., 'CEO', 'CFO')"
    )
    company: Optional[str] = Field(
        None, description="Company affiliation if detected"
    )

    # Statistics
    turn_count: int = Field(
        default=0, description="Number of times this speaker spoke"
    )
    first_appearance_page: Optional[int] = Field(
        None, description="Page where speaker first appears"
    )


class SpeakerRegistry(BaseModel):
    """Registry of all speakers identified in the transcript."""

    speakers: dict[str, SpeakerInfo] = Field(
        default_factory=dict,
        description="Map of speaker_id to SpeakerInfo"
    )

    # Convenience counts
    total_speakers: int = Field(default=0)
    management_count: int = Field(default=0)
    analyst_count: int = Field(default=0)

    def get_by_name(self, name: str) -> Optional[SpeakerInfo]:
        """Look up speaker by name or alias."""
        name_lower = name.lower().strip()
        for speaker in self.speakers.values():
            if speaker.canonical_name.lower() == name_lower:
                return speaker
            if any(alias.lower() == name_lower for alias in speaker.aliases):
                return speaker
        return None


# =============================================================================
# Stage 4: Q&A Extraction
# =============================================================================

class SpeakerTurn(BaseModel):
    """A single turn of speech by one speaker."""

    speaker_name: str = Field(description="Speaker name as it appears")
    speaker_id: Optional[str] = Field(
        None, description="Reference to speaker registry (if resolved)"
    )
    text: str = Field(description="What they said")
    page_number: int = Field(ge=1, description="Page where this turn appears")
    is_question: bool = Field(
        default=False, description="Whether this turn contains a question"
    )


class ExtractedQAUnit(BaseModel):
    """A single Q&A exchange extracted from a qa_session section.

    Contains raw extraction - no enrichment yet.

    HYBRID INTELLIGENCE:
    - LLM has full authority to decide boundaries
    - LLM provides justification for decisions
    - Follow-ups linked naturally by LLM context understanding
    """

    qa_id: str = Field(description="Unique identifier (e.g., 'qa_001')")
    source_section_id: str = Field(
        description="Reference to TranscriptSection this came from"
    )

    # Question side
    questioner_name: str = Field(description="Name of person asking")
    questioner_id: Optional[str] = Field(
        None, description="Reference to speaker registry"
    )
    question_turns: list[SpeakerTurn] = Field(
        default_factory=list,
        description="Question turn(s) - may be multiple for multi-part questions"
    )

    # Response side
    response_turns: list[SpeakerTurn] = Field(
        default_factory=list,
        description="Response turn(s) - preserves who said what"
    )
    responder_names: list[str] = Field(
        default_factory=list,
        description="Names of all responders"
    )
    responder_ids: list[str] = Field(
        default_factory=list,
        description="References to speaker registry"
    )

    # Combined text (convenience)
    question_text: str = Field(
        default="", description="Combined question text"
    )
    response_text: str = Field(
        default="", description="Combined response text"
    )

    # Follow-up tracking
    is_follow_up: bool = Field(
        default=False, description="Whether this is a follow-up question"
    )
    follow_up_of: Optional[str] = Field(
        None, description="qa_id of the parent Q&A if this is a follow-up"
    )

    # Location
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    sequence_in_session: int = Field(
        ge=0, description="Order within the Q&A session"
    )

    # LLM justification for boundary decisions (for traceability)
    boundary_justification: Optional[str] = Field(
        None, description="LLM justification for why these boundaries were chosen"
    )


class QAExtractionResult(BaseModel):
    """Complete output of Q&A extraction stage."""

    qa_units: list[ExtractedQAUnit] = Field(default_factory=list)
    total_qa_units: int = Field(default=0)
    total_follow_ups: int = Field(default=0)
    unique_questioners: int = Field(default=0)

    # Validation
    sections_processed: int = Field(default=0)
    sections_with_no_qa: list[str] = Field(
        default_factory=list,
        description="Section IDs that had no Q&A extracted (potential issues)"
    )


# =============================================================================
# Stage 5: Strategic Statement Extraction
# =============================================================================

class ExtractedStrategicStatement(BaseModel):
    """A strategic statement extracted from opening/closing remarks.

    Contains raw extraction - no enrichment yet.
    """

    statement_id: str = Field(description="Unique identifier")
    source_section_id: str = Field(
        description="Reference to TranscriptSection this came from"
    )

    # Content
    speaker_name: str = Field(description="Who made this statement")
    speaker_id: Optional[str] = Field(None, description="Reference to speaker registry")
    text: str = Field(description="The statement text")

    # Classification
    statement_type: str = Field(
        description="guidance|outlook|strategic_initiative|operational_update|financial_highlight|risk_disclosure|other"
    )
    is_forward_looking: bool = Field(
        default=False, description="Whether statement is forward-looking"
    )

    # Location
    page_number: int = Field(ge=1)
    sequence_in_section: int = Field(ge=0)


class StrategicExtractionResult(BaseModel):
    """Complete output of strategic statement extraction stage."""

    statements: list[ExtractedStrategicStatement] = Field(default_factory=list)
    total_statements: int = Field(default=0)
    forward_looking_count: int = Field(default=0)
    sections_processed: int = Field(default=0)


# =============================================================================
# Stage 6: Enrichment Models
# =============================================================================

class TopicMention(BaseModel):
    """A topic identified in content."""

    topic_name: str = Field(description="Specific topic name")
    topic_category: str = Field(
        description="Category (Financial, Operational, Strategic, etc.)"
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Direct quotes supporting this topic"
    )
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How relevant this topic is to the content"
    )


class InvestorIntent(BaseModel):
    """Analysis of investor's intent in asking a question."""

    primary_intent: str = Field(
        description="concern|clarification|validation|exploration|challenge|follow_up"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Brief explanation")


class ResponsePosture(BaseModel):
    """Analysis of management's response posture."""

    primary_posture: str = Field(
        description="confident|cautious|defensive|evasive|transparent|optimistic|neutral"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Brief explanation")


class EvidenceSpan(BaseModel):
    """A specific piece of evidence from the transcript."""

    quote: str = Field(description="Verbatim quote")
    page_number: int = Field(ge=1)
    speaker_name: Optional[str] = Field(None)
    relevance: str = Field(default="", description="Why this quote is relevant")


class EnrichedQAUnit(BaseModel):
    """Q&A unit with full enrichment applied."""

    # Base extraction (copied from ExtractedQAUnit)
    qa_id: str
    source_section_id: str
    questioner_name: str
    questioner_id: Optional[str] = None
    question_turns: list[SpeakerTurn] = Field(default_factory=list)
    response_turns: list[SpeakerTurn] = Field(default_factory=list)
    responder_names: list[str] = Field(default_factory=list)
    responder_ids: list[str] = Field(default_factory=list)
    question_text: str = ""
    response_text: str = ""
    is_follow_up: bool = False
    follow_up_of: Optional[str] = None
    start_page: int = 1
    end_page: int = 1
    sequence_in_session: int = 0
    boundary_justification: Optional[str] = None

    # Enrichment
    topics: list[TopicMention] = Field(default_factory=list)
    investor_intent: Optional[InvestorIntent] = None
    response_posture: Optional[ResponsePosture] = None
    key_evidence: list[EvidenceSpan] = Field(default_factory=list)

    # Summary
    summary: Optional[str] = Field(
        None, description="Brief summary of this Q&A exchange"
    )


class EnrichedStrategicStatement(BaseModel):
    """Strategic statement with enrichment applied."""

    # Base extraction
    statement_id: str
    source_section_id: str
    speaker_name: str
    speaker_id: Optional[str] = None
    text: str
    statement_type: str
    is_forward_looking: bool = False
    page_number: int = 1
    sequence_in_section: int = 0

    # Enrichment
    topics: list[TopicMention] = Field(default_factory=list)
    key_evidence: list[EvidenceSpan] = Field(default_factory=list)
    summary: Optional[str] = None


# =============================================================================
# Pipeline State (for orchestration)
# =============================================================================

class PipelineV2State(BaseModel):
    """Complete state passed through the v2 pipeline.

    HYBRID INTELLIGENCE:
    - Stage outputs contain the final results
    - Stage traces contain inspectable intermediate decisions
    """

    # Input
    source_file: str
    raw_text: str = ""
    total_pages: int = 0

    # Stage outputs
    metadata: Optional[ExtractedMetadata] = None
    boundary_result: Optional[BoundaryDetectionResult] = None
    speaker_registry: Optional[SpeakerRegistry] = None
    qa_extraction_result: Optional[QAExtractionResult] = None
    strategic_extraction_result: Optional[StrategicExtractionResult] = None
    enriched_qa_units: list[EnrichedQAUnit] = Field(default_factory=list)
    enriched_strategic_statements: list[EnrichedStrategicStatement] = Field(default_factory=list)

    # HYBRID INTELLIGENCE: Inspectable stage traces
    # These are stored as dicts since trace classes are dataclasses, not Pydantic models
    boundary_trace: Optional[dict] = Field(
        None, description="Inspectable trace from boundary detection"
    )
    speaker_registry_trace: Optional[dict] = Field(
        None, description="Inspectable trace from speaker registry building"
    )
    qa_extraction_trace: Optional[dict] = Field(
        None, description="Inspectable trace from Q&A extraction"
    )

    # Processing metadata
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    stage_durations: dict[str, float] = Field(default_factory=dict)
    llm_calls_made: int = 0
    errors: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)

    # Validation results
    validation_passed: bool = True
    validation_issues: list[str] = Field(default_factory=list)
