"""
Upload Route

Handles PDF file uploads with validation.
"""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import UploadResponse
from services import storage

router = APIRouter()

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF file for analysis.

    The file is saved to disk and assigned a unique ID that can be used
    to trigger analysis via the /analyze endpoint.

    Returns:
        UploadResponse with file_id and metadata
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Check PDF magic bytes
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    # Save file
    file_id, file_path = await storage.save_upload(content, file.filename)

    # Try to get page count (optional)
    page_count = None
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
    except Exception:
        pass  # Page count is optional

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=len(content),
        page_count=page_count,
        upload_time=datetime.utcnow(),
    )
