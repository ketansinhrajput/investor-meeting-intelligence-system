"""Models for PDF extraction and text chunking."""

from datetime import datetime

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Content extracted from a single PDF page."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Extracted text content")
    char_offset_start: int = Field(..., ge=0, description="Global character position start")
    char_offset_end: int = Field(..., ge=0, description="Global character position end")


class RawDocument(BaseModel):
    """Complete extracted document from PDF."""

    source_file: str = Field(..., description="Path to source PDF file")
    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When extraction occurred"
    )
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    pages: list[PageContent] = Field(..., description="Content per page")
    total_characters: int = Field(..., ge=0, description="Total character count")

    @property
    def full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n".join(page.text for page in self.pages)


class DocumentChunk(BaseModel):
    """A chunk of document text for LLM processing."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    chunk_index: int = Field(..., ge=0, description="Order index of chunk")
    text: str = Field(..., description="Chunk text content")
    start_page: int = Field(..., ge=1, description="Starting page number")
    end_page: int = Field(..., ge=1, description="Ending page number")
    char_offset_start: int = Field(..., ge=0, description="Global character position start")
    char_offset_end: int = Field(..., ge=0, description="Global character position end")
    token_count: int = Field(..., ge=0, description="Estimated token count")
    overlap_with_previous: int = Field(
        default=0, ge=0, description="Characters overlapping with previous chunk"
    )
    overlap_with_next: int = Field(
        default=0, ge=0, description="Characters overlapping with next chunk"
    )
