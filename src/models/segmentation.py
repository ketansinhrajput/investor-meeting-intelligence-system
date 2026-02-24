"""Models for conversation segmentation."""

from pydantic import BaseModel, Field

from .enums import CallPhaseType, SpeakerRole


class SpeakerTurn(BaseModel):
    """A single speaker turn in the transcript."""

    turn_id: str = Field(..., description="Unique identifier for this turn")
    speaker_name: str | None = Field(None, description="Speaker name as it appears")
    speaker_id: str = Field(..., description="Normalized speaker identifier")
    inferred_role: SpeakerRole = Field(..., description="Inferred speaker role")
    text: str = Field(..., description="Turn text content")
    start_char: int = Field(..., ge=0, description="Global character position start")
    end_char: int = Field(..., ge=0, description="Global character position end")
    page_number: int = Field(..., ge=1, description="Page where turn appears")


class CallPhase(BaseModel):
    """A detected phase of the earnings call."""

    phase_type: CallPhaseType = Field(..., description="Type of call phase")
    start_turn_id: str = Field(..., description="ID of first turn in phase")
    end_turn_id: str = Field(..., description="ID of last turn in phase")
    start_page: int = Field(..., ge=1, description="Starting page")
    end_page: int = Field(..., ge=1, description="Ending page")


class SpeakerProfile(BaseModel):
    """Profile for a unique speaker in the transcript."""

    speaker_id: str = Field(..., description="Unique speaker identifier")
    canonical_name: str = Field(..., description="Normalized speaker name")
    role: SpeakerRole = Field(..., description="Speaker role")
    title: str | None = Field(None, description="Speaker title if identified (e.g., CEO)")
    organization: str | None = Field(
        None, description="Speaker organization (e.g., Morgan Stanley)"
    )
    mention_count: int = Field(default=1, ge=1, description="Number of turns by this speaker")


class SpeakerRegistry(BaseModel):
    """Registry of all speakers in the transcript."""

    speakers: dict[str, SpeakerProfile] = Field(
        default_factory=dict, description="Speaker ID to profile mapping"
    )

    def add_speaker(self, profile: SpeakerProfile) -> None:
        """Add or update a speaker in the registry."""
        self.speakers[profile.speaker_id] = profile

    def get_speaker(self, speaker_id: str) -> SpeakerProfile | None:
        """Get speaker profile by ID."""
        return self.speakers.get(speaker_id)


class SegmentedChunk(BaseModel):
    """A segmented chunk with identified turns and phases."""

    chunk_id: str = Field(..., description="Reference to source chunk")
    turns: list[SpeakerTurn] = Field(default_factory=list, description="Speaker turns in chunk")
    detected_phases: list[CallPhase] = Field(
        default_factory=list, description="Phases detected in chunk"
    )
    continuation_from_previous: bool = Field(
        default=False, description="Whether first turn continues from previous chunk"
    )
    continues_to_next: bool = Field(
        default=False, description="Whether last turn continues to next chunk"
    )


class SegmentedTranscript(BaseModel):
    """Fully segmented transcript after chunk aggregation."""

    turns: list[SpeakerTurn] = Field(..., description="All speaker turns in order")
    phases: list[CallPhase] = Field(..., description="All detected call phases")
    speaker_registry: SpeakerRegistry = Field(..., description="Registry of all speakers")
