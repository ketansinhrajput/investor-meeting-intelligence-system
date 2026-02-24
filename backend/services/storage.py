"""
Storage Service for Run Data

Handles all file I/O for uploads and run data.
Uses JSON files on disk - no database required.

Design Decisions:
- Each run gets its own directory under data/runs/{run_id}/
- Intermediate stages saved separately for debugging
- File operations are async-friendly using aiofiles
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiofiles.os

# Base directories
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"


def generate_id() -> str:
    """Generate a unique ID for files or runs."""
    return str(uuid.uuid4())[:12]


def get_upload_path(file_id: str, filename: str) -> Path:
    """Get the path for an uploaded file."""
    return UPLOADS_DIR / f"{file_id}_{filename}"


def get_run_dir(run_id: str) -> Path:
    """Get the directory for a run."""
    return RUNS_DIR / run_id


def get_run_file(run_id: str, filename: str) -> Path:
    """Get a specific file within a run directory."""
    return get_run_dir(run_id) / filename


# =============================================================================
# Upload Operations
# =============================================================================

async def save_upload(file_content: bytes, filename: str) -> tuple[str, Path]:
    """Save an uploaded file and return its ID and path."""
    file_id = generate_id()
    file_path = get_upload_path(file_id, filename)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_content)

    return file_id, file_path


async def get_upload(file_id: str) -> Optional[Path]:
    """Get the path of an uploaded file by ID."""
    # Search for files starting with the file_id
    if not UPLOADS_DIR.exists():
        return None

    for file_path in UPLOADS_DIR.iterdir():
        if file_path.name.startswith(file_id):
            return file_path
    return None


# =============================================================================
# Run Operations
# =============================================================================

async def create_run(file_id: str, filename: str) -> str:
    """Create a new run and return its ID."""
    run_id = generate_id()
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create run metadata
    metadata = {
        "run_id": run_id,
        "file_id": file_id,
        "filename": filename,
        "status": "queued",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "stages": {},
        "errors": [],
        "warnings": [],
    }

    await save_run_file(run_id, "metadata.json", metadata)
    return run_id


async def save_run_file(run_id: str, filename: str, data: Any) -> None:
    """Save data to a file in the run directory."""
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    file_path = run_dir / filename

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=2, default=str))


async def load_run_file(run_id: str, filename: str) -> Optional[dict]:
    """Load data from a file in the run directory."""
    file_path = get_run_file(run_id, filename)

    if not file_path.exists():
        return None

    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)


async def update_run_status(
    run_id: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Update the status of a run."""
    metadata = await load_run_file(run_id, "metadata.json")
    if metadata:
        metadata["status"] = status
        if status in ("completed", "failed"):
            metadata["completed_at"] = datetime.utcnow().isoformat()
        if error_message:
            metadata["errors"].append(error_message)
        await save_run_file(run_id, "metadata.json", metadata)


async def save_stage_result(
    run_id: str,
    stage_name: str,
    result: dict,
    trace: Optional[dict] = None
) -> None:
    """Save the result and trace of a pipeline stage."""
    await save_run_file(run_id, f"stage_{stage_name}_result.json", result)
    if trace:
        await save_run_file(run_id, f"stage_{stage_name}_trace.json", trace)

    # Update metadata with stage info
    metadata = await load_run_file(run_id, "metadata.json")
    if metadata:
        if "stages" not in metadata:
            metadata["stages"] = {}
        metadata["stages"][stage_name] = {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
        }
        await save_run_file(run_id, "metadata.json", metadata)


async def list_runs() -> list[dict]:
    """List all runs with their metadata."""
    runs = []

    if not RUNS_DIR.exists():
        return runs

    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir():
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    metadata = json.loads(content)
                    runs.append(metadata)

    return runs


async def get_run_metadata(run_id: str) -> Optional[dict]:
    """Get metadata for a specific run."""
    return await load_run_file(run_id, "metadata.json")


async def run_exists(run_id: str) -> bool:
    """Check if a run exists."""
    return get_run_dir(run_id).exists()


async def delete_run(run_id: str) -> bool:
    """Delete a run and all its files."""
    run_dir = get_run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
        return True
    return False


# =============================================================================
# Result Loading Helpers
# =============================================================================

async def load_pipeline_output(run_id: str) -> Optional[dict]:
    """Load the final pipeline output."""
    return await load_run_file(run_id, "pipeline_output.json")


async def load_speakers(run_id: str) -> Optional[dict]:
    """Load speaker registry from a run."""
    return await load_run_file(run_id, "stage_speakers_result.json")


async def load_qa_units(run_id: str) -> Optional[dict]:
    """Load Q&A units from a run."""
    return await load_run_file(run_id, "stage_qa_result.json")


async def load_traces(run_id: str) -> dict:
    """Load all traces from a run."""
    traces = {}

    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        return traces

    for file_path in run_dir.iterdir():
        if file_path.name.endswith("_trace.json"):
            stage_name = file_path.name.replace("stage_", "").replace("_trace.json", "")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                traces[stage_name] = json.loads(content)

    return traces


async def load_raw_text(run_id: str) -> Optional[dict]:
    """Load raw extracted text from a run."""
    return await load_run_file(run_id, "stage_extraction_result.json")


# =============================================================================
# Display Name Generation
# =============================================================================

def generate_display_name(
    extracted_metadata: dict,
    original_filename: str,
) -> str:
    """Generate a human-friendly display name for a run.

    Priority:
        1. company + quarter + year  → "GMM Pfaudler Q2 FY26"
        2. company (confidence ≥ 0.7) → "GMM Pfaudler"
        3. original filename          → "transcript"
    """
    company = extracted_metadata.get("company_name")
    quarter = extracted_metadata.get("fiscal_quarter")  # "Q2"
    year = extracted_metadata.get("fiscal_year")         # 2026
    confidence = extracted_metadata.get("extraction_confidence", 0.0)

    if company and quarter and year:
        short_year = f"FY{str(year)[-2:]}"
        return f"{company} {quarter} {short_year}"

    if company and confidence >= 0.7:
        return company

    # Fallback: strip extension and underscores
    stem = original_filename
    for ext in (".pdf", ".PDF"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem.replace("_", " ").strip()


async def update_run_display_name(run_id: str, display_name: str) -> None:
    """Set the display_name field on a run's metadata."""
    metadata = await load_run_file(run_id, "metadata.json")
    if metadata:
        metadata["display_name"] = display_name
        await save_run_file(run_id, "metadata.json", metadata)
