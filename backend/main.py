"""
FastAPI Backend for Call Transcript Intelligence System

This is the main entry point for the API server. It provides endpoints for:
- Uploading PDFs
- Triggering analysis runs
- Retrieving results, speaker registries, Q&A units, and LLM traces

In production mode, also serves the frontend SPA from backend/static/.

Architecture Decision:
- No database - all data stored as JSON on disk for simplicity and portability
- Each run gets a unique ID and its own directory
- Intermediate pipeline stages preserved for debugging
"""

# Load .env BEFORE any application imports that read os.environ
from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import upload, analyze, results, traces, auth, chat


# Ensure data directories exist on startup
DATA_DIR = Path(__file__).parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"

# Frontend static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize directories and resources on startup."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    try:
        from database import init_db
        init_db()
    except Exception as e:
        print(f"[WARN] Could not initialize database: {e}")
        print("[WARN] Auth features will be unavailable until MySQL is configured.")

    yield
    # Cleanup if needed


app = FastAPI(
    title="Call Transcript Intelligence API",
    description="API for analyzing earnings call transcripts with LLM-powered extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow dev server and any origin when serving static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for PDF previews
if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# =============================================================================
# API routes - mounted under /api prefix for production
# =============================================================================

app.include_router(auth.router, prefix="/api", tags=["Auth"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(results.router, prefix="/api", tags=["Results"])
app.include_router(traces.router, prefix="/api", tags=["Traces"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

# Also keep routes without prefix for backward compatibility (dev proxy strips /api)
app.include_router(auth.router, tags=["Auth (compat)"], include_in_schema=False)
app.include_router(upload.router, tags=["Upload (compat)"], include_in_schema=False)
app.include_router(analyze.router, tags=["Analysis (compat)"], include_in_schema=False)
app.include_router(results.router, tags=["Results (compat)"], include_in_schema=False)
app.include_router(traces.router, tags=["Traces (compat)"], include_in_schema=False)
app.include_router(chat.router, tags=["Chat (compat)"], include_in_schema=False)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# =============================================================================
# Frontend static file serving (SPA)
# =============================================================================

if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    # Serve static assets (JS, CSS, images)
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="frontend-assets",
    )

    @app.get("/vite.svg", include_in_schema=False)
    async def vite_svg():
        return FileResponse(str(STATIC_DIR / "vite.svg"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(request: Request, full_path: str):
        """Serve the frontend SPA for all non-API routes.

        This catch-all route serves index.html for client-side routing.
        API routes (/api/*) and static assets (/assets/*) are handled above.
        """
        # If it looks like a file request, try to serve it
        file_path = STATIC_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # Otherwise serve index.html for SPA routing
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    async def root():
        """Root endpoint - shown when no frontend build is available."""
        return {
            "name": "Call Transcript Intelligence API",
            "docs": "/docs",
            "note": "No frontend build found. Run 'npm run build' in frontend/ and copy dist/ to backend/static/",
            "endpoints": {
                "upload": "POST /api/upload",
                "analyze": "POST /api/analyze",
                "runs": "GET /api/runs",
                "run_summary": "GET /api/runs/{run_id}/summary",
                "speakers": "GET /api/runs/{run_id}/speakers",
                "qa": "GET /api/runs/{run_id}/qa",
                "traces": "GET /api/runs/{run_id}/traces",
            }
        }


# =============================================================================
# Run with: python main.py
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8100))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
