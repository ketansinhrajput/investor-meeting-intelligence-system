"""Semantic-aware text chunking for LLM processing."""

import re
import uuid
from dataclasses import dataclass

import structlog
import tiktoken

from src.models import DocumentChunk, RawDocument

logger = structlog.get_logger(__name__)

# Speaker pattern for detecting speaker turns (e.g., "John Smith:" or "CEO:")
SPEAKER_PATTERN = re.compile(r"^([A-Z][a-zA-Z\s\.\-\']+):\s*", re.MULTILINE)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    target_tokens: int = 6000  # ~24000 characters
    overlap_tokens: int = 500  # ~2000 characters
    min_chunk_tokens: int = 1000
    max_chunk_tokens: int = 7500
    encoding_name: str = "cl100k_base"  # GPT-4 encoding, reasonable default


def chunk_document(
    doc: RawDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Chunk a document for LLM processing.

    Uses semantic-aware splitting to:
    - Respect paragraph boundaries
    - Preserve speaker turn boundaries where possible
    - Maintain overlap between chunks for context continuity

    Args:
        doc: Raw document to chunk.
        config: Chunking configuration.

    Returns:
        List of document chunks.
    """
    config = config or ChunkingConfig()

    # Get tokenizer
    try:
        encoding = tiktoken.get_encoding(config.encoding_name)
    except Exception:
        logger.warning("tiktoken_encoding_fallback", requested=config.encoding_name)
        encoding = tiktoken.get_encoding("cl100k_base")

    # Concatenate all pages with page markers for tracking
    full_text, page_boundaries = _build_full_text_with_boundaries(doc)

    logger.info(
        "chunking_document",
        total_chars=len(full_text),
        target_tokens=config.target_tokens,
    )

    # Find all split points
    split_points = _find_split_points(full_text)

    # Build chunks using split points
    chunks = _build_chunks(
        full_text=full_text,
        page_boundaries=page_boundaries,
        split_points=split_points,
        encoding=encoding,
        config=config,
    )

    logger.info("chunking_complete", num_chunks=len(chunks))

    return chunks


def _build_full_text_with_boundaries(
    doc: RawDocument,
) -> tuple[str, list[tuple[int, int, int]]]:
    """Build full text and track page boundaries.

    Returns:
        Tuple of (full_text, page_boundaries) where page_boundaries
        is a list of (page_num, start_offset, end_offset).
    """
    text_parts = []
    page_boundaries = []
    current_offset = 0

    for page in doc.pages:
        start = current_offset
        text_parts.append(page.text)
        current_offset += len(page.text)
        end = current_offset

        page_boundaries.append((page.page_number, start, end))

        # Add newline separator between pages
        if page != doc.pages[-1]:
            text_parts.append("\n")
            current_offset += 1

    return "".join(text_parts), page_boundaries


def _find_split_points(text: str) -> list[tuple[int, int]]:
    """Find potential split points in text with priority scores.

    Returns list of (position, priority) tuples.
    Higher priority = better split point.

    Priority levels:
    - 100: Paragraph boundary (double newline)
    - 90: Speaker turn boundary
    - 50: Sentence boundary
    - 20: Clause boundary (comma, semicolon)
    """
    split_points: list[tuple[int, int]] = []

    # Paragraph boundaries (highest priority)
    for match in re.finditer(r"\n\n", text):
        split_points.append((match.end(), 100))

    # Speaker turn boundaries
    for match in SPEAKER_PATTERN.finditer(text):
        # Split before the speaker name
        split_points.append((match.start(), 90))

    # Sentence boundaries
    for match in re.finditer(r"[.!?]\s+(?=[A-Z])", text):
        split_points.append((match.end(), 50))

    # Clause boundaries
    for match in re.finditer(r"[,;]\s+", text):
        split_points.append((match.end(), 20))

    # Sort by position
    split_points.sort(key=lambda x: x[0])

    return split_points


def _build_chunks(
    full_text: str,
    page_boundaries: list[tuple[int, int, int]],
    split_points: list[tuple[int, int]],
    encoding: tiktoken.Encoding,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    """Build chunks from text using split points.

    Args:
        full_text: Complete document text.
        page_boundaries: List of (page_num, start, end) tuples.
        split_points: List of (position, priority) tuples.
        encoding: Tiktoken encoding for token counting.
        config: Chunking configuration.

    Returns:
        List of DocumentChunk objects.
    """
    chunks: list[DocumentChunk] = []
    current_start = 0
    chunk_index = 0

    while current_start < len(full_text):
        # Find the best end point for this chunk
        target_end = _find_chunk_end(
            text=full_text,
            start=current_start,
            split_points=split_points,
            encoding=encoding,
            config=config,
        )

        # Extract chunk text
        chunk_text = full_text[current_start:target_end].strip()

        if not chunk_text:
            break

        # Count tokens
        token_count = len(encoding.encode(chunk_text))

        # Find page range
        start_page, end_page = _get_page_range(
            current_start, target_end, page_boundaries
        )

        # Calculate overlap with previous chunk
        overlap_prev = 0
        if chunks:
            prev_chunk = chunks[-1]
            overlap_prev = prev_chunk.char_offset_end - current_start
            if overlap_prev < 0:
                overlap_prev = 0

        chunk = DocumentChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
            chunk_index=chunk_index,
            text=chunk_text,
            start_page=start_page,
            end_page=end_page,
            char_offset_start=current_start,
            char_offset_end=target_end,
            token_count=token_count,
            overlap_with_previous=overlap_prev,
        )

        # Update previous chunk's overlap_with_next
        if chunks:
            chunks[-1].overlap_with_next = overlap_prev

        chunks.append(chunk)

        logger.debug(
            "chunk_created",
            index=chunk_index,
            tokens=token_count,
            pages=f"{start_page}-{end_page}",
        )

        # Move to next chunk start with overlap
        overlap_chars = _tokens_to_chars(config.overlap_tokens)
        next_start = target_end - overlap_chars

        # Ensure we make progress
        if next_start <= current_start:
            next_start = target_end

        # Find a good split point near the overlap boundary
        next_start = _find_nearest_split_point(
            next_start, split_points, full_text, direction="forward"
        )

        current_start = min(next_start, len(full_text))
        chunk_index += 1

    return chunks


def _find_chunk_end(
    text: str,
    start: int,
    split_points: list[tuple[int, int]],
    encoding: tiktoken.Encoding,
    config: ChunkingConfig,
) -> int:
    """Find optimal end position for a chunk starting at given position.

    Tries to get as close to target_tokens as possible while
    respecting split point priorities.
    """
    # Estimate character count for target tokens
    target_chars = _tokens_to_chars(config.target_tokens)
    max_chars = _tokens_to_chars(config.max_chunk_tokens)

    ideal_end = min(start + target_chars, len(text))
    max_end = min(start + max_chars, len(text))

    # If we can fit the rest of the text, just return it
    remaining_text = text[start:]
    remaining_tokens = len(encoding.encode(remaining_text))
    if remaining_tokens <= config.max_chunk_tokens:
        return len(text)

    # Find split points in the acceptable range
    # Look for points between 80% and 100% of target, preferring higher priority
    min_acceptable = start + int(target_chars * 0.8)
    candidates = [
        (pos, priority)
        for pos, priority in split_points
        if min_acceptable <= pos <= max_end
    ]

    if candidates:
        # Sort by priority (descending), then by closeness to ideal (ascending)
        candidates.sort(key=lambda x: (-x[1], abs(x[0] - ideal_end)))
        return candidates[0][0]

    # No good split point found, try to find any split point before max_end
    fallback_candidates = [
        (pos, priority)
        for pos, priority in split_points
        if start < pos <= max_end
    ]

    if fallback_candidates:
        fallback_candidates.sort(key=lambda x: (-x[1], -x[0]))  # Highest priority, latest position
        return fallback_candidates[0][0]

    # Last resort: just cut at max_end
    return max_end


def _find_nearest_split_point(
    position: int,
    split_points: list[tuple[int, int]],
    text: str,
    direction: str = "forward",
    max_distance: int = 500,
) -> int:
    """Find nearest good split point to a position.

    Args:
        position: Target position.
        split_points: Available split points.
        text: Full text (for bounds checking).
        direction: 'forward' or 'backward'.
        max_distance: Maximum distance to search.

    Returns:
        Best split point position, or original position if none found.
    """
    if direction == "forward":
        candidates = [
            (pos, priority)
            for pos, priority in split_points
            if position <= pos <= position + max_distance
        ]
    else:
        candidates = [
            (pos, priority)
            for pos, priority in split_points
            if position - max_distance <= pos <= position
        ]

    if candidates:
        # Prefer higher priority, then closer to position
        candidates.sort(key=lambda x: (-x[1], abs(x[0] - position)))
        return candidates[0][0]

    return min(position, len(text))


def _get_page_range(
    start_offset: int,
    end_offset: int,
    page_boundaries: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Get page range for a character offset range.

    Returns:
        Tuple of (start_page, end_page).
    """
    start_page = 1
    end_page = 1

    for page_num, page_start, page_end in page_boundaries:
        if page_start <= start_offset < page_end:
            start_page = page_num
        if page_start < end_offset <= page_end:
            end_page = page_num
        elif end_offset > page_end:
            end_page = page_num

    return start_page, end_page


def _tokens_to_chars(tokens: int) -> int:
    """Estimate character count from token count.

    Uses rough estimate of 4 characters per token.
    """
    return tokens * 4


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text.

    Args:
        text: Text to count.
        encoding_name: Tiktoken encoding name.

    Returns:
        Token count.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimate
        return len(text) // 4
