"""LangGraph workflow definition for the transcript analysis pipeline."""

import structlog
from langgraph.graph import END, START, StateGraph

from src.pipeline.nodes import (
    aggregate_chunks_node,
    chunk_document_node,
    enrich_units_node,
    extract_pdf_node,
    extract_qa_units_node,
    extract_strategic_statements_node,
    generate_report_node,
    segment_chunks_node,
)
from src.pipeline.state import PipelineState, create_initial_state

logger = structlog.get_logger(__name__)


def should_continue_after_extraction(state: PipelineState) -> str:
    """Determine if processing should continue after PDF extraction.

    Args:
        state: Current pipeline state.

    Returns:
        'continue' if extraction succeeded, 'error' otherwise.
    """
    if state.get("raw_document") is None:
        logger.warning("pipeline_stopping_no_document")
        return "error"
    return "continue"


def should_continue_after_chunking(state: PipelineState) -> str:
    """Determine if processing should continue after chunking.

    Args:
        state: Current pipeline state.

    Returns:
        'continue' if chunks exist, 'error' otherwise.
    """
    if not state.get("chunks"):
        logger.warning("pipeline_stopping_no_chunks")
        return "error"
    return "continue"


def should_continue_after_segmentation(state: PipelineState) -> str:
    """Determine if processing should continue after segmentation.

    Args:
        state: Current pipeline state.

    Returns:
        'continue' if segments exist, 'error' otherwise.
    """
    if not state.get("segmented_chunks"):
        logger.warning("pipeline_stopping_no_segments")
        return "error"
    return "continue"


def should_continue_after_aggregation(state: PipelineState) -> str:
    """Determine if processing should continue after aggregation.

    Args:
        state: Current pipeline state.

    Returns:
        'continue' if transcript exists, 'error' otherwise.
    """
    if not state.get("segmented_transcript"):
        logger.warning("pipeline_stopping_no_transcript")
        return "error"
    return "continue"


def build_pipeline() -> StateGraph:
    """Build the LangGraph workflow for transcript processing.

    Returns:
        Compiled StateGraph ready for execution.
    """
    logger.info("building_pipeline")

    # Initialize graph with state schema
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("pdf_extraction", extract_pdf_node)
    workflow.add_node("chunking", chunk_document_node)
    workflow.add_node("segmentation", segment_chunks_node)
    workflow.add_node("aggregation", aggregate_chunks_node)
    workflow.add_node("qa_extraction", extract_qa_units_node)
    workflow.add_node("strategic_extraction", extract_strategic_statements_node)
    workflow.add_node("enrichment", enrich_units_node)
    workflow.add_node("report_generation", generate_report_node)

    # Define edges

    # Start -> PDF Extraction
    workflow.add_edge(START, "pdf_extraction")

    # PDF Extraction -> Chunking (conditional on success)
    workflow.add_conditional_edges(
        "pdf_extraction",
        should_continue_after_extraction,
        {
            "continue": "chunking",
            "error": "report_generation",
        },
    )

    # Chunking -> Segmentation (conditional)
    workflow.add_conditional_edges(
        "chunking",
        should_continue_after_chunking,
        {
            "continue": "segmentation",
            "error": "report_generation",
        },
    )

    # Segmentation -> Aggregation (conditional)
    workflow.add_conditional_edges(
        "segmentation",
        should_continue_after_segmentation,
        {
            "continue": "aggregation",
            "error": "report_generation",
        },
    )

    # Aggregation -> Q&A Extraction and Strategic Extraction (parallel)
    # Note: LangGraph doesn't directly support parallel execution from one node
    # We'll run them sequentially for simplicity, but they could be parallelized
    # using asyncio or separate subgraphs
    workflow.add_conditional_edges(
        "aggregation",
        should_continue_after_aggregation,
        {
            "continue": "qa_extraction",
            "error": "report_generation",
        },
    )

    # Q&A Extraction -> Strategic Extraction
    workflow.add_edge("qa_extraction", "strategic_extraction")

    # Strategic Extraction -> Enrichment
    workflow.add_edge("strategic_extraction", "enrichment")

    # Enrichment -> Report Generation
    workflow.add_edge("enrichment", "report_generation")

    # Report Generation -> End
    workflow.add_edge("report_generation", END)

    logger.info("pipeline_built")

    return workflow


def create_pipeline_app():
    """Create compiled pipeline application.

    Returns:
        Compiled LangGraph application.
    """
    workflow = build_pipeline()
    return workflow.compile()


def run_pipeline(pdf_path: str) -> dict:
    """Execute the full pipeline on a transcript PDF.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        Final report dict.
    """
    logger.info("pipeline_starting", pdf_path=pdf_path)

    # Create pipeline
    app = create_pipeline_app()

    # Create initial state
    initial_state = create_initial_state(pdf_path)

    # Run pipeline
    result = app.invoke(initial_state)

    logger.info("pipeline_complete")

    return result.get("report", {})


async def run_pipeline_async(pdf_path: str) -> dict:
    """Execute the pipeline asynchronously.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        Final report dict.
    """
    logger.info("pipeline_starting_async", pdf_path=pdf_path)

    # Create pipeline
    app = create_pipeline_app()

    # Create initial state
    initial_state = create_initial_state(pdf_path)

    # Run pipeline asynchronously
    result = await app.ainvoke(initial_state)

    logger.info("pipeline_complete_async")

    return result.get("report", {})
