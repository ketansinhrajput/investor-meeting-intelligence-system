"""
Analyze Route

Triggers pipeline execution on uploaded files.
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import AnalyzeRequest, AnalyzeResponse
from services import storage, pipeline_runner

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """
    Start analysis of an uploaded PDF.

    The analysis runs in the background. Use the returned run_id
    to check status and retrieve results.

    Args:
        request: Contains file_id of the uploaded file

    Returns:
        AnalyzeResponse with run_id for tracking
    """
    # Find the uploaded file
    file_path = await storage.get_upload(request.file_id)

    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.file_id}"
        )

    # Extract filename
    filename = file_path.name.split("_", 1)[-1] if "_" in file_path.name else file_path.name

    # Create run record
    run_id = await storage.create_run(request.file_id, filename)

    # Start pipeline in background
    background_tasks.add_task(
        pipeline_runner.run_pipeline,
        run_id,
        file_path,
        request.skip_enrichment,
    )

    return AnalyzeResponse(
        run_id=run_id,
        file_id=request.file_id,
        status="queued",
        started_at=datetime.utcnow(),
    )
