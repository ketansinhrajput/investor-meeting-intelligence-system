"""Models for strategic statement extraction and enrichment."""

from pydantic import BaseModel, Field

from .enums import CallPhaseType, SentimentType, StatementType
from .enrichment import EvidenceReference, InferredTopic


class StrategicStatement(BaseModel):
    """A strategic statement extracted from the transcript."""

    statement_id: str = Field(..., description="Unique identifier")
    speaker_id: str = Field(..., description="Speaker ID")
    speaker_name: str = Field(..., description="Speaker name")
    speaker_title: str | None = Field(None, description="Speaker title if known")

    phase: CallPhaseType = Field(..., description="Call phase where statement occurred")
    text: str = Field(..., description="Full statement text")
    turn_ids: list[str] = Field(..., description="Turn IDs containing this statement")
    statement_type: StatementType = Field(..., description="Type of strategic statement")

    start_page: int = Field(..., ge=1, description="Starting page")
    end_page: int = Field(..., ge=1, description="Ending page")


class EnrichedStrategicStatement(BaseModel):
    """A strategic statement with analytical enrichment."""

    statement_id: str = Field(..., description="Unique identifier")

    # Content
    text: str = Field(..., description="Full statement text")
    summary: str = Field(..., description="Brief 1-2 sentence summary")

    # Enrichment
    topics: list[InferredTopic] = Field(default_factory=list, description="Inferred topics")
    sentiment: SentimentType = Field(..., description="Overall sentiment")
    forward_looking: bool = Field(..., description="Whether statement is forward-looking")

    # Evidence
    key_evidence: list[EvidenceReference] = Field(
        default_factory=list, description="Supporting evidence"
    )

    # Original metadata
    speaker_id: str = Field(..., description="Speaker ID")
    speaker_name: str = Field(..., description="Speaker name")
    speaker_title: str | None = Field(None, description="Speaker title")
    statement_type: StatementType = Field(..., description="Statement type")
    start_page: int = Field(..., ge=1, description="Starting page")
    end_page: int = Field(..., ge=1, description="Ending page")
