"""Models for Q&A unit extraction and enrichment."""

from pydantic import BaseModel, Field

from .enrichment import EvidenceReference, InferredTopic, InvestorIntent, ResponsePosture


class QAUnit(BaseModel):
    """A Q&A exchange unit (question + response)."""

    unit_id: str = Field(..., description="Unique identifier for this Q&A unit")
    sequence_number: int = Field(..., ge=1, description="Order in Q&A session")

    # Question component
    question_turns: list[str] = Field(..., description="Turn IDs containing the question")
    questioner_id: str = Field(..., description="Speaker ID of questioner")
    questioner_name: str | None = Field(None, description="Name of questioner")
    questioner_organization: str | None = Field(
        None, description="Organization of questioner (e.g., Goldman Sachs)"
    )

    # Response component
    response_turns: list[str] = Field(..., description="Turn IDs containing responses")
    responders: list[str] = Field(..., description="Speaker IDs of responders")

    # Location
    start_page: int = Field(..., ge=1, description="Starting page")
    end_page: int = Field(..., ge=1, description="Ending page")

    # Context
    moderator_introduction: str | None = Field(
        None, description="Moderator intro text if present"
    )
    moderator_turn_id: str | None = Field(None, description="Turn ID of moderator intro")


class EnrichedQAUnit(BaseModel):
    """A Q&A unit with analytical enrichment."""

    unit_id: str = Field(..., description="Unique identifier")
    sequence_number: int = Field(..., ge=1, description="Order in Q&A session")

    # Verbatim content
    question_text: str = Field(..., description="Full question text")
    response_text: str = Field(..., description="Full response text")

    # Enrichment
    topics: list[InferredTopic] = Field(default_factory=list, description="Inferred topics")
    investor_intent: InvestorIntent = Field(..., description="Inferred investor intent")
    response_posture: ResponsePosture = Field(..., description="Inferred response posture")

    # Evidence
    key_evidence: list[EvidenceReference] = Field(
        default_factory=list, description="Supporting evidence"
    )

    # Original metadata
    questioner_id: str = Field(..., description="Speaker ID of questioner")
    questioner_name: str | None = Field(None, description="Name of questioner")
    questioner_organization: str | None = Field(None, description="Organization of questioner")
    responders: list[str] = Field(..., description="Speaker IDs of responders")
    start_page: int = Field(..., ge=1, description="Starting page")
    end_page: int = Field(..., ge=1, description="Ending page")
