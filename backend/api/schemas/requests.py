"""
Request schemas for the API.

These define the expected input structure for API endpoints.
Using Pydantic v2 for validation and serialization.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request to analyze a previously uploaded PDF."""
    file_id: str = Field(..., description="ID of the uploaded file to analyze")
    skip_enrichment: bool = Field(default=False, description="Skip enrichment stages for faster processing")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"file_id": "abc123", "skip_enrichment": False}
            ]
        }
    }


class LoginRequest(BaseModel):
    """Request to log in with username and password."""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class ChatMessage(BaseModel):
    """A single message in chat history."""
    role: str = Field(..., pattern="^(user|assistant)$", description="Message role: user or assistant")
    content: str = Field(..., min_length=1, description="Message content")


class ChatRequest(BaseModel):
    """Request to chat about a specific analysis run."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Previous messages in this conversation (for context)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What was discussed about EBITDA margins?",
                    "history": [],
                }
            ]
        }
    }


class RerunStageRequest(BaseModel):
    """Request to re-run a specific pipeline stage.

    Future feature: allows re-running individual stages with modified parameters.
    """
    run_id: str = Field(..., description="ID of the run to modify")
    stage: str = Field(..., description="Stage to re-run (boundary, speakers, qa, strategic)")
    parameters: Optional[dict] = Field(default=None, description="Override parameters for the stage")
