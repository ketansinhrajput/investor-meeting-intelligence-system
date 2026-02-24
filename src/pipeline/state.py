"""Pipeline state definition for LangGraph."""

from typing import Annotated, TypedDict

import operator


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline.

    Uses Annotated types with operators for list accumulation.
    """

    # Input
    pdf_path: str

    # Stage 1: PDF Extraction
    raw_document: dict | None  # RawDocument as dict

    # Stage 2: Chunking
    chunks: Annotated[list[dict], operator.add]  # List[DocumentChunk]

    # Stage 3: Segmentation (per-chunk results)
    segmented_chunks: Annotated[list[dict], operator.add]  # List[SegmentedChunk]

    # Stage 4: Aggregated transcript
    segmented_transcript: dict | None  # SegmentedTranscript
    speaker_registry: dict  # SpeakerRegistry

    # Stage 5: Q&A Units
    qa_units: Annotated[list[dict], operator.add]  # List[QAUnit]

    # Stage 6: Strategic Statements
    strategic_statements: Annotated[list[dict], operator.add]  # List[StrategicStatement]

    # Stage 7: Enriched units
    enriched_qa_units: Annotated[list[dict], operator.add]  # List[EnrichedQAUnit]
    enriched_strategic: Annotated[list[dict], operator.add]  # List[EnrichedStrategicStatement]

    # Stage 8-9: Final report
    topic_summaries: list[dict]  # List[TopicSummary]
    call_metadata: dict | None  # CallMetadata
    report: dict | None  # StructuredReport

    # Processing metadata
    processing_start: str  # ISO timestamp
    llm_calls_count: int
    chunks_count: int

    # Error tracking
    errors: Annotated[list[dict], operator.add]  # List[ProcessingError]
    warnings: Annotated[list[dict], operator.add]  # List[ProcessingError]


def create_initial_state(pdf_path: str) -> PipelineState:
    """Create initial pipeline state.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        Initial PipelineState dict.
    """
    from datetime import datetime

    return PipelineState(
        pdf_path=pdf_path,
        raw_document=None,
        chunks=[],
        segmented_chunks=[],
        segmented_transcript=None,
        speaker_registry={},
        qa_units=[],
        strategic_statements=[],
        enriched_qa_units=[],
        enriched_strategic=[],
        topic_summaries=[],
        call_metadata=None,
        report=None,
        processing_start=datetime.utcnow().isoformat(),
        llm_calls_count=0,
        chunks_count=0,
        errors=[],
        warnings=[],
    )
