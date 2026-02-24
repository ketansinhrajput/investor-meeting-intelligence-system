"""
Response schemas for the API.

These define the output structure for API endpoints.
All schemas are designed to be self-describing for frontend consumption.

Key Design Decisions:
- Every list includes count for pagination preparation
- All IDs are strings for flexibility
- Traces include full context for debugging
- Speaker and QA data include cross-references
"""

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Auth Response
# =============================================================================

class LoginResponse(BaseModel):
    """Response after successful login."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer")
    username: str
    role: str


# =============================================================================
# Upload Response
# =============================================================================

class UploadResponse(BaseModel):
    """Response after uploading a PDF file."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages if PDF parsed")
    upload_time: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Analysis Response
# =============================================================================

class AnalyzeResponse(BaseModel):
    """Response after starting analysis."""
    run_id: str = Field(..., description="Unique identifier for this analysis run")
    file_id: str = Field(..., description="ID of the file being analyzed")
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    started_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Run List Response
# =============================================================================

class RunListItem(BaseModel):
    """Summary of a single run for listing."""
    run_id: str
    file_id: str
    filename: str
    display_name: Optional[str] = None
    status: Literal["queued", "running", "completed", "failed"]
    started_at: datetime
    completed_at: Optional[datetime] = None
    qa_count: Optional[int] = None
    speaker_count: Optional[int] = None
    error_message: Optional[str] = None


class RunListResponse(BaseModel):
    """List of all runs."""
    runs: list[RunListItem]
    total_count: int


# =============================================================================
# Run Summary Response
# =============================================================================

class PipelineStageStatus(BaseModel):
    """Status of a single pipeline stage."""
    stage_name: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class RunSummaryResponse(BaseModel):
    """High-level summary of a run."""
    run_id: str
    file_id: str
    filename: str
    display_name: Optional[str] = None
    status: Literal["queued", "running", "completed", "failed"]

    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Document stats
    page_count: int = 0
    total_text_length: int = 0

    # Extraction stats
    speaker_count: int = 0
    qa_count: int = 0
    follow_up_count: int = 0
    strategic_statement_count: int = 0

    # Pipeline stages
    stages: list[PipelineStageStatus] = Field(default_factory=list)

    # Errors and warnings
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None


# =============================================================================
# Speaker Registry Response
# =============================================================================

class SpeakerAlias(BaseModel):
    """An alias for a speaker."""
    alias: str
    merge_reason: Optional[str] = None
    confidence: Optional[float] = None


class SpeakerResponse(BaseModel):
    """A single speaker in the registry."""
    speaker_id: str
    canonical_name: str
    role: Literal["moderator", "management", "analyst", "unknown"]
    title: Optional[str] = None
    company: Optional[str] = None
    turn_count: int = 0
    first_appearance_page: Optional[int] = None
    aliases: list[SpeakerAlias] = Field(default_factory=list)

    # LLM decision info (for debugging)
    verified_by_llm: bool = False
    llm_confidence: Optional[float] = None
    llm_reasoning: Optional[str] = None


class SpeakerRegistryResponse(BaseModel):
    """Full speaker registry for a run."""
    run_id: str
    speakers: list[SpeakerResponse]
    total_count: int
    management_count: int = 0
    analyst_count: int = 0
    moderator_count: int = 0


# =============================================================================
# Q&A Response
# =============================================================================

class SpeakerTurnResponse(BaseModel):
    """A single speaker turn within a Q&A unit."""
    speaker_name: str
    speaker_id: Optional[str] = None
    text: str
    page_number: Optional[int] = None
    is_question: bool = False


class QAUnitResponse(BaseModel):
    """A single Q&A unit."""
    qa_id: str
    sequence: int = 0  # Order in session

    # Question info
    questioner_name: str
    questioner_id: Optional[str] = None
    questioner_company: Optional[str] = None
    question_text: str
    question_turns: list[SpeakerTurnResponse] = Field(default_factory=list)

    # Response info
    responder_names: list[str] = Field(default_factory=list)
    responder_ids: list[str] = Field(default_factory=list)
    response_text: str = ""
    response_turns: list[SpeakerTurnResponse] = Field(default_factory=list)

    # Follow-up chain
    is_follow_up: bool = False
    follow_up_of: Optional[str] = None
    has_follow_ups: bool = False
    follow_up_ids: list[str] = Field(default_factory=list)

    # Location
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    source_section_id: Optional[str] = None

    # Enrichment data
    topics: list[str] = Field(default_factory=list)
    investor_intent: Optional[str] = None  # clarification, concern, exploration, challenge
    response_posture: Optional[str] = None  # confident, cautious, defensive, optimistic, neutral

    # LLM decision info
    boundary_reasoning: Optional[str] = None
    confidence: Optional[float] = None


class QAResponse(BaseModel):
    """Full Q&A extraction for a run."""
    run_id: str
    qa_units: list[QAUnitResponse]
    total_count: int
    follow_up_count: int = 0
    unique_questioners: int = 0


# =============================================================================
# Traces Response
# =============================================================================

class EvidenceSpanResponse(BaseModel):
    """Evidence supporting an LLM decision."""
    text: str
    source: str
    relevance: str


class TraceDecision(BaseModel):
    """A single LLM decision within a trace."""
    decision_id: str
    decision_type: str  # e.g., "speaker_verification", "qa_boundary", "role_assignment"
    input_context: str  # What the LLM saw
    output_decision: str  # What the LLM decided
    confidence: Optional[float] = None
    reasoning: str = ""
    evidence_spans: list[EvidenceSpanResponse] = Field(default_factory=list)
    timestamp: Optional[datetime] = None


class StageTrace(BaseModel):
    """Trace for a single pipeline stage."""
    stage_name: str
    stage_type: str  # e.g., "boundary_detection", "speaker_registry", "qa_extraction"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    llm_calls_made: int = 0
    decisions: list[TraceDecision] = Field(default_factory=list)
    hard_rules_enforced: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TracesResponse(BaseModel):
    """All traces for a run."""
    run_id: str
    stages: list[StageTrace]
    total_llm_calls: int = 0


# =============================================================================
# Raw Text Response
# =============================================================================

class PageText(BaseModel):
    """Raw text for a single page."""
    page_number: int
    text: str
    char_count: int = 0


class RawTextResponse(BaseModel):
    """Raw extracted text per page."""
    run_id: str
    pages: list[PageText]
    total_pages: int
    total_chars: int = 0


# =============================================================================
# Raw JSON Response (for debugging)
# =============================================================================

class RawJsonResponse(BaseModel):
    """Raw pipeline output as JSON."""
    run_id: str
    pipeline_output: dict[str, Any]
    stages_output: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Chat Response
# =============================================================================

class ChatCitation(BaseModel):
    """A structured citation in a chat response."""
    type: Literal["qa", "speaker", "page"] = Field(description="What is being cited")
    ref_id: str = Field(description="ID of the cited item")
    label: str = Field(description="Human-readable label")


class ChatToolCall(BaseModel):
    """A tool call made during chat processing."""
    tool: str = Field(description="Tool name that was called")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters passed to the tool")


class ChatResponse(BaseModel):
    """Response from the chat agent."""
    run_id: str
    answer: str = Field(description="The agent's grounded response")
    citations: list[ChatCitation] = Field(default_factory=list)
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    retrieval_source: str = Field(description="Where evidence came from: structured, vector, both, or none")
    total_time_seconds: float = Field(description="Total processing time")
    model: str = Field(description="LLM model used")
    disclaimer: str = Field(default="", description="Caveat about the answer source")
