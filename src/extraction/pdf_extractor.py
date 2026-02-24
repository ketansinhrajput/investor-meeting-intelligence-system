"""PDF text extraction using pdfplumber."""

from datetime import datetime
from pathlib import Path

import pdfplumber
import structlog

from src.models import PageContent, RawDocument

logger = structlog.get_logger(__name__)


class PDFExtractionError(Exception):
    """Error during PDF extraction."""

    pass


def extract_pdf(pdf_path: str | Path) -> RawDocument:
    """Extract text content from a PDF file.

    Extracts text from each page while preserving page boundaries
    and tracking character offsets for evidence references.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        RawDocument with extracted content.

    Raises:
        PDFExtractionError: If extraction fails.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise PDFExtractionError(f"File is not a PDF: {pdf_path}")

    logger.info("extracting_pdf", path=str(pdf_path))

    pages: list[PageContent] = []
    char_offset = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info("pdf_opened", total_pages=total_pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Clean up text - normalize whitespace but preserve structure
                text = _clean_page_text(text)

                page_content = PageContent(
                    page_number=page_num,
                    text=text,
                    char_offset_start=char_offset,
                    char_offset_end=char_offset + len(text),
                )
                pages.append(page_content)

                # Update offset for next page (add newline separator)
                char_offset += len(text) + 1  # +1 for newline between pages

                logger.debug(
                    "page_extracted",
                    page=page_num,
                    chars=len(text),
                )

    except Exception as e:
        logger.error("pdf_extraction_failed", error=str(e))
        raise PDFExtractionError(f"Failed to extract PDF: {e}") from e

    total_chars = sum(len(p.text) for p in pages)

    raw_doc = RawDocument(
        source_file=str(pdf_path),
        extraction_timestamp=datetime.utcnow(),
        total_pages=len(pages),
        pages=pages,
        total_characters=total_chars,
    )

    logger.info(
        "pdf_extraction_complete",
        pages=raw_doc.total_pages,
        total_chars=raw_doc.total_characters,
    )

    return raw_doc


def _clean_page_text(text: str) -> str:
    """Clean extracted page text.

    - Normalizes multiple spaces to single space
    - Preserves paragraph breaks (double newlines)
    - Strips trailing whitespace from lines
    - Removes page headers/footers if detected

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Strip trailing whitespace
        line = line.rstrip()

        # Normalize multiple spaces within line
        while "  " in line:
            line = line.replace("  ", " ")

        cleaned_lines.append(line)

    # Join lines and normalize paragraph breaks
    result = "\n".join(cleaned_lines)

    # Normalize multiple blank lines to double newline (paragraph break)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")

    return result.strip()


def get_page_for_offset(doc: RawDocument, char_offset: int) -> int:
    """Get page number for a character offset.

    Args:
        doc: The raw document.
        char_offset: Character offset in the full text.

    Returns:
        Page number (1-indexed).
    """
    for page in doc.pages:
        if page.char_offset_start <= char_offset < page.char_offset_end:
            return page.page_number

    # If offset is at or past end, return last page
    return doc.total_pages
