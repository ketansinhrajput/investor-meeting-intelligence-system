"""Pipeline V2 Orchestrator - Coordinates all pipeline stages.

Philosophy: "Recall and structure first, interpretation later."

HYBRID INTELLIGENCE:
- Each stage combines deterministic structure with LLM-assisted decisions
- All stages return inspectable traces for auditability
- Traces are stored in PipelineV2State for downstream analysis

This orchestrator runs all stages in sequence, maintaining state
and handling errors gracefully.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from src.extraction.pdf_extractor import extract_pdf
from src.pipeline_v2.models import PipelineV2State
from src.pipeline_v2.stages import (
    build_speaker_registry,
    detect_boundaries,
    enrich_qa_units,
    enrich_strategic_statements,
    extract_metadata,
    extract_qa_units,
    extract_strategic_statements,
)
from src.pipeline_v2.stages.text_cleaner import clean_extracted_text
from src.pipeline_v2.stages.qa_block_processor import process_all_qa_sections

logger = structlog.get_logger(__name__)


class PipelineV2Error(Exception):
    """Error during pipeline v2 execution."""
    pass


def run_pipeline_v2(
    pdf_path: str | Path,
    skip_enrichment: bool = False,
    max_qa_enrichment: Optional[int] = None,
    max_strategic_enrichment: Optional[int] = None,
    use_block_processor: bool = True,
) -> PipelineV2State:
    """Run the complete v2 pipeline on a PDF transcript.

    HYBRID APPROACH:
    - Regex proposes (high recall)
    - LLM labels and structures (semantic understanding)
    - Nothing is silently dropped

    Args:
        pdf_path: Path to the PDF transcript file.
        skip_enrichment: If True, skip the enrichment stage (faster but less detail).
        max_qa_enrichment: Limit number of Q&A units to enrich (for testing).
        max_strategic_enrichment: Limit number of strategic statements to enrich.
        use_block_processor: If True, use block-by-block Q&A extraction (recommended).

    Returns:
        PipelineV2State with all stage outputs.

    Raises:
        PipelineV2Error: If a critical stage fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PipelineV2Error(f"PDF file not found: {pdf_path}")

    # Initialize state
    state = PipelineV2State(
        source_file=str(pdf_path),
        processing_start=datetime.now(),
    )

    logger.info("pipeline_v2_start", source=str(pdf_path))

    try:
        # Stage 0: PDF Extraction
        state = _run_pdf_extraction(state, pdf_path)

        # Stage 0B: Text Cleaning (regex noise removal)
        state = _run_text_cleaning(state)

        # Stage 1: Metadata Extraction
        state = _run_metadata_extraction(state)

        # Stage 2: Boundary Detection
        state = _run_boundary_detection(state)

        # Stage 3: Speaker Registry
        state = _run_speaker_registry(state)

        # Stage 4: Q&A Extraction
        if use_block_processor:
            state = _run_qa_block_extraction(state)
        else:
            state = _run_qa_extraction(state)

        # Stage 5: Strategic Extraction
        state = _run_strategic_extraction(state)

        # Stage 6: Enrichment (optional)
        if not skip_enrichment:
            state = _run_enrichment(state, max_qa_enrichment, max_strategic_enrichment)

        # Finalize
        state.processing_end = datetime.now()
        state.validation_passed = _validate_state(state)

        total_duration = (state.processing_end - state.processing_start).total_seconds()
        logger.info(
            "pipeline_v2_complete",
            duration_seconds=round(total_duration, 2),
            qa_units=len(state.enriched_qa_units) if state.enriched_qa_units else (
                state.qa_extraction_result.total_qa_units if state.qa_extraction_result else 0
            ),
            strategic_statements=len(state.enriched_strategic_statements) if state.enriched_strategic_statements else (
                state.strategic_extraction_result.total_statements if state.strategic_extraction_result else 0
            ),
            llm_calls=state.llm_calls_made,
            validation_passed=state.validation_passed,
        )

        return state

    except Exception as e:
        state.processing_end = datetime.now()
        state.errors.append({
            "stage": "orchestrator",
            "error": str(e),
            "type": type(e).__name__,
        })
        logger.error("pipeline_v2_failed", error=str(e))
        raise PipelineV2Error(f"Pipeline failed: {e}") from e


def _run_pdf_extraction(state: PipelineV2State, pdf_path: Path) -> PipelineV2State:
    """Stage 0: Extract text from PDF."""
    stage_start = datetime.now()
    logger.info("stage_0_pdf_extraction_start")

    try:
        raw_doc = extract_pdf(pdf_path)
        state.raw_text = raw_doc.full_text
        state.total_pages = raw_doc.total_pages

        logger.info(
            "stage_0_complete",
            pages=state.total_pages,
            text_length=len(state.raw_text),
        )

    except Exception as e:
        state.errors.append({"stage": "pdf_extraction", "error": str(e)})
        logger.error("stage_0_failed", error=str(e))
        raise

    state.stage_durations["pdf_extraction"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_text_cleaning(state: PipelineV2State) -> PipelineV2State:
    """Stage 0B: Clean extracted text (regex noise removal).

    HYBRID APPROACH: Regex removes OBVIOUS noise before LLM sees text.
    - Page headers (Company Name + Date)
    - Page markers (Page X of Y)
    - Repeated earnings call titles
    """
    stage_start = datetime.now()
    logger.info("stage_0b_text_cleaning_start")

    try:
        company_name = state.metadata.company_name if state.metadata else None
        cleaned = clean_extracted_text(
            raw_text=state.raw_text,
            company_name=company_name,
        )

        # Update state with cleaned text
        original_length = len(state.raw_text)
        state.raw_text = cleaned.text

        logger.info(
            "stage_0b_complete",
            original_length=original_length,
            cleaned_length=len(state.raw_text),
            chars_removed=cleaned.trace.chars_removed,
            page_markers_removed=len(cleaned.trace.page_markers_removed),
            headers_removed=len(cleaned.trace.page_headers_removed),
        )

    except Exception as e:
        state.warnings.append({"stage": "text_cleaning", "error": str(e)})
        logger.warning("stage_0b_partial_failure", error=str(e))
        # Non-critical - continue with uncleaned text

    state.stage_durations["text_cleaning"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_metadata_extraction(state: PipelineV2State) -> PipelineV2State:
    """Stage 1: Extract call metadata."""
    stage_start = datetime.now()
    logger.info("stage_1_metadata_start")

    try:
        state.metadata = extract_metadata(
            full_text=state.raw_text,
            use_llm=True,
        )
        state.llm_calls_made += 1

        logger.info(
            "stage_1_complete",
            company=state.metadata.company_name,
            quarter=state.metadata.fiscal_quarter,
            year=state.metadata.fiscal_year,
        )

    except Exception as e:
        state.warnings.append({"stage": "metadata_extraction", "error": str(e)})
        logger.warning("stage_1_partial_failure", error=str(e))
        # Non-critical - continue with None metadata

    state.stage_durations["metadata_extraction"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_boundary_detection(state: PipelineV2State) -> PipelineV2State:
    """Stage 2: Detect section boundaries (HYBRID INTELLIGENCE)."""
    stage_start = datetime.now()
    logger.info("stage_2_boundary_start")

    try:
        # HYBRID: detect_boundaries returns (result, trace) tuple
        result = detect_boundaries(
            full_text=state.raw_text,
            total_pages=state.total_pages,
            use_llm_confirmation=True,
        )

        # Handle both old (single return) and new (tuple return) signatures
        if isinstance(result, tuple):
            state.boundary_result, trace = result
            # Store trace as dict for serialization
            state.boundary_trace = asdict(trace)
            state.llm_calls_made += len([c for c in trace.candidates if c.llm_confirmed is not None])
        else:
            state.boundary_result = result

        # 🔧 NORMALIZE sequence numbers (must be >= 0 and contiguous)
        for idx, section in enumerate(state.boundary_result.sections):
            section.sequence_number = idx

        logger.info(
            "stage_2_complete",
            sections=state.boundary_result.total_sections,
            qa_sections=state.boundary_result.qa_section_count,
            coverage=round(state.boundary_result.coverage_percent, 1),
            trace_candidates=len(state.boundary_trace.get("candidates", [])) if state.boundary_trace else 0,
        )

        # Validation warning if coverage is low
        if state.boundary_result.coverage_percent < 90:
            state.warnings.append({
                "stage": "boundary_detection",
                "warning": f"Low coverage: {state.boundary_result.coverage_percent:.1f}%",
            })

    except Exception as e:
        state.errors.append({"stage": "boundary_detection", "error": str(e)})
        logger.error("stage_2_failed", error=str(e))
        raise

    state.stage_durations["boundary_detection"] = (
        datetime.now() - stage_start
    ).total_seconds()

    return state



def _run_speaker_registry(state: PipelineV2State) -> PipelineV2State:
    """Stage 3: Build speaker registry (HYBRID INTELLIGENCE)."""
    stage_start = datetime.now()
    logger.info("stage_3_speakers_start")

    try:
        # HYBRID: build_speaker_registry returns (result, trace) tuple
        result = build_speaker_registry(
            boundary_result=state.boundary_result,
            metadata=state.metadata,
            full_text=state.raw_text,
            use_llm=True,
        )

        # Handle both old (single return) and new (tuple return) signatures
        if isinstance(result, tuple):
            state.speaker_registry, trace = result
            # Store trace as dict for serialization
            state.speaker_registry_trace = asdict(trace)
            state.llm_calls_made += trace.llm_calls_made
        else:
            state.speaker_registry = result

        logger.info(
            "stage_3_complete",
            total_speakers=state.speaker_registry.total_speakers,
            management=state.speaker_registry.management_count,
            analysts=state.speaker_registry.analyst_count,
            trace_decisions=len(state.speaker_registry_trace.get("role_decisions", [])) if state.speaker_registry_trace else 0,
        )

    except Exception as e:
        state.warnings.append({"stage": "speaker_registry", "error": str(e)})
        logger.warning("stage_3_partial_failure", error=str(e))
        # Non-critical - continue without registry

    state.stage_durations["speaker_registry"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_qa_block_extraction(state: PipelineV2State) -> PipelineV2State:
    """Stage 4: Extract Q&A units using block-by-block processing.

    HYBRID APPROACH:
    - Regex detects Q&A blocks (investor introductions)
    - LLM structures each block internally
    - LLM MUST return a result for every block (never silent drop)
    """
    stage_start = datetime.now()
    logger.info("stage_4_qa_block_extraction_start")

    try:
        from dataclasses import asdict

        result, trace = process_all_qa_sections(
            boundary_result=state.boundary_result,
            full_text=state.raw_text,
            total_pages=state.total_pages,
            registry=state.speaker_registry,
        )

        state.qa_extraction_result = result
        state.qa_extraction_trace = asdict(trace)
        state.llm_calls_made += trace.llm_calls

        logger.info(
            "stage_4_complete",
            blocks_detected=trace.blocks_detected,
            blocks_valid=trace.blocks_valid,
            blocks_invalid=trace.blocks_invalid,
            qa_units=result.total_qa_units,
            follow_ups=result.total_follow_ups,
        )

        # Validation warning if no Q&A found
        if result.total_qa_units == 0:
            state.warnings.append({
                "stage": "qa_block_extraction",
                "warning": "No Q&A units extracted",
                "invalid_blocks": trace.invalid_reasons,
            })

    except Exception as e:
        state.errors.append({"stage": "qa_block_extraction", "error": str(e)})
        logger.error("stage_4_failed", error=str(e))
        raise

    state.stage_durations["qa_extraction"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_qa_extraction(state: PipelineV2State) -> PipelineV2State:
    """Stage 4: Extract Q&A units (legacy approach)."""
    stage_start = datetime.now()
    logger.info("stage_4_qa_extraction_start")

    try:
        # HYBRID: extract_qa_units returns (result, trace) tuple
        result = extract_qa_units(
            boundary_result=state.boundary_result,
            registry=state.speaker_registry,
            use_llm_fallback=True,
        )

        # Handle both old (single return) and new (tuple return) signatures
        if isinstance(result, tuple):
            state.qa_extraction_result, trace = result
            # Store trace as dict for serialization
            state.qa_extraction_trace = asdict(trace)
            state.llm_calls_made += trace.llm_calls_made
        else:
            state.qa_extraction_result = result
            # Fallback for old signature
            llm_calls = len(state.qa_extraction_result.sections_with_no_qa)
            state.llm_calls_made += llm_calls

        logger.info(
            "stage_4_complete",
            qa_units=state.qa_extraction_result.total_qa_units,
            follow_ups=state.qa_extraction_result.total_follow_ups,
            unique_questioners=state.qa_extraction_result.unique_questioners,
            trace_turns=len(state.qa_extraction_trace.get("turn_candidates", [])) if state.qa_extraction_trace else 0,
        )

        # Validation warning if no Q&A found
        if state.qa_extraction_result.total_qa_units == 0:
            state.warnings.append({
                "stage": "qa_extraction",
                "warning": "No Q&A units extracted",
            })

    except Exception as e:
        state.errors.append({"stage": "qa_extraction", "error": str(e)})
        logger.error("stage_4_failed", error=str(e))
        raise

    state.stage_durations["qa_extraction"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_strategic_extraction(state: PipelineV2State) -> PipelineV2State:
    """Stage 5: Extract strategic statements."""
    stage_start = datetime.now()
    logger.info("stage_5_strategic_start")

    try:
        state.strategic_extraction_result = extract_strategic_statements(
            boundary_result=state.boundary_result,
            registry=state.speaker_registry,
            use_llm=True,
        )

        state.llm_calls_made += state.strategic_extraction_result.sections_processed

        logger.info(
            "stage_5_complete",
            statements=state.strategic_extraction_result.total_statements,
            forward_looking=state.strategic_extraction_result.forward_looking_count,
        )

    except Exception as e:
        state.warnings.append({"stage": "strategic_extraction", "error": str(e)})
        logger.warning("stage_5_partial_failure", error=str(e))
        # Non-critical - continue without strategic statements

    state.stage_durations["strategic_extraction"] = (datetime.now() - stage_start).total_seconds()
    return state


def _run_enrichment(
    state: PipelineV2State,
    max_qa: Optional[int],
    max_strategic: Optional[int],
) -> PipelineV2State:
    """Stage 6: Enrich Q&A units and strategic statements."""
    stage_start = datetime.now()
    logger.info("stage_6_enrichment_start")

    try:
        # Enrich Q&A units
        if state.qa_extraction_result and state.qa_extraction_result.qa_units:
            state.enriched_qa_units = enrich_qa_units(
                qa_result=state.qa_extraction_result,
                max_units=max_qa,
            )
            state.llm_calls_made += len(state.enriched_qa_units)

        # Enrich strategic statements
        if state.strategic_extraction_result and state.strategic_extraction_result.statements:
            state.enriched_strategic_statements = enrich_strategic_statements(
                strategic_result=state.strategic_extraction_result,
                max_statements=max_strategic,
            )
            state.llm_calls_made += len(state.enriched_strategic_statements)

        logger.info(
            "stage_6_complete",
            enriched_qa=len(state.enriched_qa_units),
            enriched_strategic=len(state.enriched_strategic_statements),
        )

    except Exception as e:
        state.warnings.append({"stage": "enrichment", "error": str(e)})
        logger.warning("stage_6_partial_failure", error=str(e))
        # Non-critical - continue with unenriched data

    state.stage_durations["enrichment"] = (datetime.now() - stage_start).total_seconds()
    return state


def _validate_state(state: PipelineV2State) -> bool:
    """Validate final pipeline state."""
    issues = []

    # Check for critical data
    if not state.raw_text:
        issues.append("No text extracted from PDF")

    if not state.boundary_result or state.boundary_result.total_sections == 0:
        issues.append("No sections detected")

    if state.boundary_result and state.boundary_result.coverage_percent < 80:
        issues.append(f"Low section coverage: {state.boundary_result.coverage_percent:.1f}%")

    if not state.qa_extraction_result or state.qa_extraction_result.total_qa_units == 0:
        issues.append("No Q&A units extracted")

    state.validation_issues = issues

    if issues:
        logger.warning("validation_issues", issues=issues)

    return len(issues) == 0
