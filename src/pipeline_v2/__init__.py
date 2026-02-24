"""Pipeline v2 - Multi-stage deterministic pipeline with LLM-assisted steps.

Philosophy: "Recall and structure first, interpretation later."

Key differences from v1:
- Boundary detection ensures no Q&A is missed (high recall)
- Each stage has focused responsibility
- Smaller context windows per LLM call
- Inspectable intermediate outputs
- Non-LLM validation gates

Usage:
    from src.pipeline_v2 import run_pipeline_v2

    state = run_pipeline_v2("transcript.pdf")
    print(f"Found {len(state.enriched_qa_units)} Q&A units")
"""

from src.pipeline_v2.orchestrator import run_pipeline_v2, PipelineV2Error
from src.pipeline_v2.models import (
    # Enums
    SectionType,
    SpeakerRole,
    DetectionSignal,
    # Stage 1: Metadata
    ExtractedMetadata,
    # Stage 2: Boundary Detection
    TranscriptSection,
    BoundaryDetectionResult,
    # Stage 3: Speaker Registry
    SpeakerInfo,
    SpeakerRegistry,
    # Stage 4: Q&A Extraction
    SpeakerTurn,
    ExtractedQAUnit,
    QAExtractionResult,
    # Stage 5: Strategic Extraction
    ExtractedStrategicStatement,
    StrategicExtractionResult,
    # Stage 6: Enrichment
    TopicMention,
    InvestorIntent,
    ResponsePosture,
    EvidenceSpan,
    EnrichedQAUnit,
    EnrichedStrategicStatement,
    # Pipeline State
    PipelineV2State,
)

__all__ = [
    # Pipeline Entry Points
    "run_pipeline_v2",
    "PipelineV2Error",
    # Enums
    "SectionType",
    "SpeakerRole",
    "DetectionSignal",
    # Stage 1
    "ExtractedMetadata",
    # Stage 2
    "TranscriptSection",
    "BoundaryDetectionResult",
    # Stage 3
    "SpeakerInfo",
    "SpeakerRegistry",
    # Stage 4
    "SpeakerTurn",
    "ExtractedQAUnit",
    "QAExtractionResult",
    # Stage 5
    "ExtractedStrategicStatement",
    "StrategicExtractionResult",
    # Stage 6
    "TopicMention",
    "InvestorIntent",
    "ResponsePosture",
    "EvidenceSpan",
    "EnrichedQAUnit",
    "EnrichedStrategicStatement",
    # Pipeline State
    "PipelineV2State",
]
