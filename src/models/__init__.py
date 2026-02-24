"""Pydantic data models for the pipeline."""

from .enums import (
    CallPhaseType,
    ErrorSeverity,
    InvestorIntentType,
    ResponsePostureType,
    SentimentType,
    SpeakerRole,
    StatementType,
)
from .extraction import DocumentChunk, PageContent, RawDocument
from .segmentation import (
    CallPhase,
    SegmentedChunk,
    SegmentedTranscript,
    SpeakerProfile,
    SpeakerRegistry,
    SpeakerTurn,
)
from .qa_units import EnrichedQAUnit, QAUnit
from .strategic import EnrichedStrategicStatement, StrategicStatement
from .enrichment import EvidenceReference, InferredTopic, InvestorIntent, ResponsePosture
from .report import (
    CallMetadata,
    ProcessingError,
    ProcessingMetadata,
    StructuredReport,
    TopicSummary,
)

__all__ = [
    # Enums
    "SpeakerRole",
    "CallPhaseType",
    "InvestorIntentType",
    "ResponsePostureType",
    "StatementType",
    "SentimentType",
    "ErrorSeverity",
    # Extraction
    "RawDocument",
    "PageContent",
    "DocumentChunk",
    # Segmentation
    "SpeakerTurn",
    "CallPhase",
    "SpeakerProfile",
    "SpeakerRegistry",
    "SegmentedChunk",
    "SegmentedTranscript",
    # Q&A
    "QAUnit",
    "EnrichedQAUnit",
    # Strategic
    "StrategicStatement",
    "EnrichedStrategicStatement",
    # Enrichment
    "InferredTopic",
    "InvestorIntent",
    "ResponsePosture",
    "EvidenceReference",
    # Report
    "CallMetadata",
    "ProcessingMetadata",
    "ProcessingError",
    "TopicSummary",
    "StructuredReport",
]
