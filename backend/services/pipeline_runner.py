"""
Pipeline Runner Service

Orchestrates the execution of the transcript analysis pipeline.
Bridges the existing pipeline_v2 code with the web API.

Design Decisions:
- Runs pipeline in background to not block API
- Saves intermediate results for each stage
- Updates metadata after each stage for live frontend updates
- Captures all traces for debugging
- Converts pipeline models to JSON-serializable dicts
"""

import asyncio
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services import storage


def _to_dict(obj: Any) -> Any:
    """Convert dataclasses and Pydantic models to dicts recursively."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


async def _update_stage_status(
    run_id: str,
    stage_name: str,
    status: str,
    error: str | None = None
) -> None:
    """Update a specific stage's status in the run metadata."""
    metadata = await storage.get_run_metadata(run_id)
    if metadata:
        if "stages" not in metadata:
            metadata["stages"] = {}
        metadata["stages"][stage_name] = {
            "status": status,
            "completed_at": datetime.utcnow().isoformat() if status in ("completed", "failed") else None,
            "error": error,
        }
        await storage.save_run_file(run_id, "metadata.json", metadata)


async def run_pipeline(
    run_id: str,
    file_path: Path,
    skip_enrichment: bool = False
) -> None:
    """
    Run the full analysis pipeline on a PDF file.

    This function:
    1. Updates run status to 'running'
    2. Executes each pipeline stage
    3. Saves intermediate results and traces
    4. Updates run status to 'completed' or 'failed'

    Args:
        run_id: Unique identifier for this run
        file_path: Path to the PDF file
        skip_enrichment: Whether to skip enrichment stages for faster processing
    """
    try:
        await storage.update_run_status(run_id, "running")

        # Import pipeline modules
        try:
            from src.pipeline_v2.orchestrator import run_pipeline_v2
            pipeline_available = True
        except ImportError as e:
            pipeline_available = False
            import_error = str(e)

        if not pipeline_available:
            # Create mock result for demo/development
            await _create_mock_result(run_id, file_path, import_error)
            return

        # Run the actual pipeline
        await _run_actual_pipeline(run_id, file_path, skip_enrichment)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        print(f"Pipeline error for run {run_id}:\n{tb}")
        await storage.update_run_status(run_id, "failed", error_msg)


async def _run_actual_pipeline(
    run_id: str,
    file_path: Path,
    skip_enrichment: bool
) -> None:
    """Run the actual pipeline and save results with live updates."""
    from src.pipeline_v2.orchestrator import run_pipeline_v2

    # Mark extraction as running
    await _update_stage_status(run_id, "extraction", "running")

    # Run pipeline (this is sync, so run in executor)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline_v2(str(file_path), skip_enrichment=skip_enrichment)
        )
    except Exception as e:
        await _update_stage_status(run_id, "extraction", "failed", str(e))
        raise

    # Save extraction stage (raw text)
    if result.raw_text:
        # The pipeline doesn't give us page-by-page breakdown directly,
        # but we can create a simple representation
        extraction_result = {
            "pages": [
                {
                    "page_number": 1,
                    "text": result.raw_text[:10000] if len(result.raw_text) > 10000 else result.raw_text,
                    "char_count": len(result.raw_text),
                }
            ],
            "total_pages": result.total_pages,
            "total_chars": len(result.raw_text),
        }
        await storage.save_run_file(run_id, "stage_extraction_result.json", extraction_result)
        await _update_stage_status(run_id, "extraction", "completed")

        # Update page count immediately
        metadata = await storage.get_run_metadata(run_id)
        if metadata:
            metadata["page_count"] = result.total_pages
            await storage.save_run_file(run_id, "metadata.json", metadata)

    # Save metadata stage
    if result.metadata:
        await _update_stage_status(run_id, "metadata", "completed")
        metadata_dict = _to_dict(result.metadata)
        await storage.save_stage_result(
            run_id,
            "metadata",
            metadata_dict,
        )

        # Generate display_name from extracted metadata
        run_meta = await storage.get_run_metadata(run_id)
        if run_meta:
            original_filename = run_meta.get("filename", "transcript.pdf")
            display_name = storage.generate_display_name(metadata_dict, original_filename)
            await storage.update_run_display_name(run_id, display_name)

    # Save boundary stage
    if result.boundary_result:
        boundary_result = _to_dict(result.boundary_result)
        boundary_trace = _to_dict(result.boundary_trace) if result.boundary_trace else None
        await storage.save_stage_result(run_id, "boundary", boundary_result, boundary_trace)
        await _update_stage_status(run_id, "boundary", "completed")

    # Save speakers stage
    if result.speaker_registry:
        speakers_result = _to_dict(result.speaker_registry)
        speakers_trace = _to_dict(result.speaker_registry_trace) if result.speaker_registry_trace else None
        await storage.save_stage_result(run_id, "speakers", speakers_result, speakers_trace)
        await _update_stage_status(run_id, "speakers", "completed")

        # Update speaker count immediately
        metadata = await storage.get_run_metadata(run_id)
        if metadata:
            metadata["speaker_count"] = len(result.speaker_registry.speakers)
            await storage.save_run_file(run_id, "metadata.json", metadata)

    # Save Q&A stage
    if result.qa_extraction_result:
        qa_result = _to_dict(result.qa_extraction_result)
        qa_trace = _to_dict(result.qa_extraction_trace) if result.qa_extraction_trace else None
        await storage.save_stage_result(run_id, "qa", qa_result, qa_trace)
        await _update_stage_status(run_id, "qa", "completed")

        # Update Q&A count immediately
        metadata = await storage.get_run_metadata(run_id)
        if metadata:
            metadata["qa_count"] = len(result.qa_extraction_result.qa_units)
            await storage.save_run_file(run_id, "metadata.json", metadata)

    # Save strategic statements stage
    if result.strategic_extraction_result:
        await storage.save_stage_result(
            run_id,
            "strategic",
            _to_dict(result.strategic_extraction_result),
        )
        await _update_stage_status(run_id, "strategic", "completed")

    # Save final output
    final_output = _to_dict(result)
    await storage.save_run_file(run_id, "pipeline_output.json", final_output)

    # Final metadata update
    metadata = await storage.get_run_metadata(run_id)
    if metadata:
        metadata["status"] = "completed"
        metadata["completed_at"] = datetime.utcnow().isoformat()

        # Ensure counts are set
        if result.speaker_registry:
            metadata["speaker_count"] = len(result.speaker_registry.speakers)
        if result.qa_extraction_result:
            metadata["qa_count"] = len(result.qa_extraction_result.qa_units)
        metadata["page_count"] = result.total_pages

        # Add any errors/warnings from pipeline
        if result.errors:
            metadata["errors"] = [str(e) for e in result.errors]
        if result.warnings:
            metadata["warnings"] = [str(w) for w in result.warnings]

        await storage.save_run_file(run_id, "metadata.json", metadata)

    await storage.update_run_status(run_id, "completed")


async def _create_mock_result(run_id: str, file_path: Path, error: str) -> None:
    """Create mock results when pipeline is not available."""
    # This is useful for frontend development without the full pipeline

    await _update_stage_status(run_id, "extraction", "running")
    await asyncio.sleep(0.5)  # Simulate some work

    mock_extraction = {
        "pages": [
            {"page_number": 1, "text": "Mock page 1 content - TechCorp Q2 2024 Earnings Call...", "char_count": 500},
            {"page_number": 2, "text": "Mock page 2 content - CEO remarks and financial highlights...", "char_count": 500},
        ],
        "total_pages": 2,
        "total_chars": 1000,
    }
    await storage.save_run_file(run_id, "stage_extraction_result.json", mock_extraction)
    await _update_stage_status(run_id, "extraction", "completed")

    # Update page count
    metadata = await storage.get_run_metadata(run_id)
    if metadata:
        metadata["page_count"] = 2
        await storage.save_run_file(run_id, "metadata.json", metadata)

    await _update_stage_status(run_id, "speakers", "running")
    await asyncio.sleep(0.5)

    mock_speakers = {
        "speakers": {
            "speaker_001": {
                "speaker_id": "speaker_001",
                "canonical_name": "John Smith",
                "role": "management",
                "title": "CEO",
                "company": "Example Corp",
                "turn_count": 5,
                "aliases": [],
            },
            "speaker_002": {
                "speaker_id": "speaker_002",
                "canonical_name": "Jane Analyst",
                "role": "analyst",
                "title": None,
                "company": "Investment Bank",
                "turn_count": 3,
                "aliases": [],
            },
        },
        "total_speakers": 2,
        "management_count": 1,
        "analyst_count": 1,
    }
    await storage.save_stage_result(run_id, "speakers", mock_speakers, {
        "llm_calls_made": 2,
        "verification_decisions": [],
    })
    await _update_stage_status(run_id, "speakers", "completed")

    # Update speaker count
    metadata = await storage.get_run_metadata(run_id)
    if metadata:
        metadata["speaker_count"] = 2
        await storage.save_run_file(run_id, "metadata.json", metadata)

    await _update_stage_status(run_id, "qa", "running")
    await asyncio.sleep(0.5)

    mock_qa = {
        "qa_units": [
            {
                "qa_id": "qa_001",
                "questioner_name": "Jane Analyst",
                "question_text": "What is your guidance for next quarter?",
                "responder_names": ["John Smith"],
                "response_text": "We expect strong growth driven by our enterprise segment...",
                "is_follow_up": False,
                "start_page": 1,
                "end_page": 1,
            }
        ],
        "total_qa_units": 1,
        "total_follow_ups": 0,
    }
    await storage.save_stage_result(run_id, "qa", mock_qa, {
        "llm_calls_made": 1,
        "qa_constructions": [],
    })
    await _update_stage_status(run_id, "qa", "completed")

    # Final update
    metadata = await storage.get_run_metadata(run_id)
    if metadata:
        metadata["status"] = "completed"
        metadata["completed_at"] = datetime.utcnow().isoformat()
        metadata["speaker_count"] = 2
        metadata["qa_count"] = 1
        metadata["page_count"] = 2
        metadata["warnings"] = [f"Pipeline not available: {error}. Using mock data."]
        await storage.save_run_file(run_id, "metadata.json", metadata)

    await storage.update_run_status(run_id, "completed")
