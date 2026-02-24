"""Models for the final structured report."""

from datetime import datetime

from pydantic import BaseModel, Field

from .enums import ErrorSeverity
from .qa_units import EnrichedQAUnit
from .segmentation import CallPhase, SpeakerRegistry
from .strategic import EnrichedStrategicStatement


class CallMetadata(BaseModel):
    """Metadata about the earnings call."""

    company_name: str | None = Field(None, description="Company name if extracted")
    ticker_symbol: str | None = Field(None, description="Stock ticker if extracted")
    fiscal_quarter: str | None = Field(None, description="Fiscal quarter (Q1-Q4)")
    fiscal_year: int | None = Field(None, description="Fiscal year")
    call_date: datetime | None = Field(None, description="Date of the call")

    source_file: str = Field(..., description="Source PDF filename")
    extraction_timestamp: datetime = Field(..., description="When extraction occurred")
    total_pages: int = Field(..., ge=1, description="Total pages in transcript")

    participant_count: int = Field(default=0, ge=0, description="Number of participants")
    qa_unit_count: int = Field(default=0, ge=0, description="Number of Q&A units")
    strategic_statement_count: int = Field(
        default=0, ge=0, description="Number of strategic statements"
    )


class ProcessingError(BaseModel):
    """Record of a processing error or warning."""

    error_id: str = Field(..., description="Unique error identifier")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    stage: str = Field(..., description="Pipeline stage where error occurred")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")
    recoverable: bool = Field(default=True, description="Whether processing continued")


class ProcessingMetadata(BaseModel):
    """Metadata about the processing run."""

    pipeline_version: str = Field(..., description="Version of the pipeline")
    processing_start: datetime = Field(..., description="When processing started")
    processing_end: datetime = Field(..., description="When processing completed")
    total_duration_seconds: float = Field(..., ge=0, description="Total processing time")

    llm_model: str = Field(..., description="LLM model used")
    llm_temperature: float = Field(..., ge=0, le=1, description="LLM temperature setting")

    chunks_processed: int = Field(default=0, ge=0, description="Number of chunks processed")
    llm_calls_made: int = Field(default=0, ge=0, description="Number of LLM API calls")

    errors: list[ProcessingError] = Field(default_factory=list, description="Processing errors")
    warnings: list[ProcessingError] = Field(default_factory=list, description="Processing warnings")


class TopicSummary(BaseModel):
    """Aggregated summary for a topic across the transcript."""

    topic_name: str = Field(..., description="Topic name")
    topic_category: str = Field(..., description="Topic category")
    mention_count: int = Field(..., ge=1, description="Number of mentions")

    qa_unit_ids: list[str] = Field(
        default_factory=list, description="Q&A units discussing this topic"
    )
    statement_ids: list[str] = Field(
        default_factory=list, description="Strategic statements with this topic"
    )

    summary: str = Field(..., description="Summary of topic discussion")
    key_points: list[str] = Field(default_factory=list, description="Key points about topic")
    sentiment_distribution: dict[str, int] = Field(
        default_factory=dict, description="Sentiment counts"
    )


class StructuredReport(BaseModel):
    """The final structured report output."""

    report_id: str = Field(..., description="Unique report identifier")
    report_version: str = Field(default="1.0.0", description="Report schema version")

    # Metadata
    call_metadata: CallMetadata = Field(..., description="Call metadata")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")

    # Content
    speaker_registry: SpeakerRegistry = Field(..., description="All identified speakers")
    qa_units: list[EnrichedQAUnit] = Field(default_factory=list, description="Q&A units")
    strategic_statements: list[EnrichedStrategicStatement] = Field(
        default_factory=list, description="Strategic statements"
    )

    # Summaries
    topic_summaries: list[TopicSummary] = Field(
        default_factory=list, description="Topic-level summaries"
    )
    call_phases: list[CallPhase] = Field(default_factory=list, description="Detected call phases")
