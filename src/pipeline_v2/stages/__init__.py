"""Pipeline v2 stages - each stage has focused responsibility.

CONTEXT-AWARE HYBRID INTELLIGENCE APPROACH:
- Phase A: Deterministic candidate generation with HIGH RECALL
- Phase B: Document/Section-level LLM verification (LLM sees ALL context)
- Code enforces HARD RULES after LLM decisions

KEY PRINCIPLES:
- High recall first, precision second
- LLM has FULL AUTHORITY to restructure outputs
- Every LLM decision emits justification trace with evidence_spans
- All stages return inspectable traces for auditability
"""

from src.pipeline_v2.stages.metadata import extract_metadata
from src.pipeline_v2.stages.boundary import (
    detect_boundaries,
    BoundaryDetectionTrace,
    BoundaryCandidate,
)
from src.pipeline_v2.stages.speakers import (
    build_speaker_registry,
    SpeakerRegistryTrace,
    NameCandidate,
    SpeakerVerificationRecord,
    AliasMergeDecision,
    TitleAssignmentRecord,
)
from src.pipeline_v2.stages.qa_extraction import (
    extract_qa_units,
    QAExtractionTrace,
    TurnIntentRecord,
    QAUnitConstruction,
)
from src.pipeline_v2.stages.strategic import extract_strategic_statements
from src.pipeline_v2.stages.enrichment import enrich_qa_units, enrich_strategic_statements

__all__ = [
    # Stage functions
    "extract_metadata",
    "detect_boundaries",
    "build_speaker_registry",
    "extract_qa_units",
    "extract_strategic_statements",
    "enrich_qa_units",
    "enrich_strategic_statements",
    # Trace classes (inspectable outputs)
    "BoundaryDetectionTrace",
    "BoundaryCandidate",
    "SpeakerRegistryTrace",
    "NameCandidate",
    "SpeakerVerificationRecord",
    "AliasMergeDecision",
    "TitleAssignmentRecord",
    "QAExtractionTrace",
    "TurnIntentRecord",
    "QAUnitConstruction",
]
