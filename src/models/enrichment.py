"""Models for analytical enrichment."""

from pydantic import BaseModel, Field

from .enums import InvestorIntentType, ResponsePostureType


class InferredTopic(BaseModel):
    """A dynamically inferred discussion topic."""

    topic_name: str = Field(..., description="Specific topic name")
    topic_category: str = Field(..., description="Topic category (e.g., Financial, Operational)")
    evidence_spans: list[str] = Field(
        default_factory=list, description="Direct quotes supporting this topic"
    )


class InvestorIntent(BaseModel):
    """Inferred intent behind an investor question."""

    primary_intent: InvestorIntentType = Field(..., description="Primary intent classification")
    reasoning: str = Field(..., description="Explanation for intent classification")


class ResponsePosture(BaseModel):
    """Inferred posture of management response."""

    primary_posture: ResponsePostureType = Field(
        ..., description="Primary posture classification"
    )
    reasoning: str = Field(..., description="Explanation for posture classification")


class EvidenceReference(BaseModel):
    """A reference to source text as evidence."""

    quote: str = Field(..., description="Verbatim quote from transcript")
    speaker_id: str = Field(..., description="ID of speaker who said this")
    page_number: int = Field(..., ge=1, description="Page number where quote appears")
    char_offset_start: int | None = Field(
        None, ge=0, description="Character offset start (optional)"
    )
    char_offset_end: int | None = Field(None, ge=0, description="Character offset end (optional)")
