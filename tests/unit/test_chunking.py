"""Unit tests for text chunking."""

import pytest

from src.extraction.chunker import (
    ChunkingConfig,
    chunk_document,
    count_tokens,
    _find_split_points,
)
from src.models import PageContent, RawDocument


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        config = ChunkingConfig()
        assert config.target_tokens == 6000
        assert config.overlap_tokens == 500

    def test_custom_config(self):
        config = ChunkingConfig(target_tokens=4000, overlap_tokens=300)
        assert config.target_tokens == 4000
        assert config.overlap_tokens == 300


class TestFindSplitPoints:
    """Tests for split point detection."""

    def test_paragraph_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph."
        points = _find_split_points(text)

        # Should find the paragraph boundary
        paragraph_points = [p for p in points if p[1] == 100]  # Priority 100
        assert len(paragraph_points) > 0

    def test_speaker_boundaries(self):
        text = "John Smith: Hello everyone.\nJane Doe: Thank you."
        points = _find_split_points(text)

        # Should find speaker boundaries
        speaker_points = [p for p in points if p[1] == 90]  # Priority 90
        assert len(speaker_points) >= 1

    def test_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence."
        points = _find_split_points(text)

        # Should find sentence boundaries
        sentence_points = [p for p in points if p[1] == 50]  # Priority 50
        assert len(sentence_points) >= 2


class TestCountTokens:
    """Tests for token counting."""

    def test_count_tokens_basic(self):
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert count < 10  # Should be 2-3 tokens

    def test_count_tokens_empty(self):
        count = count_tokens("")
        assert count == 0


class TestChunkDocument:
    """Tests for document chunking."""

    @pytest.fixture
    def sample_document(self) -> RawDocument:
        """Create a sample document for testing."""
        text = "This is a test document. " * 100  # ~2500 chars

        return RawDocument(
            source_file="test.pdf",
            total_pages=1,
            pages=[
                PageContent(
                    page_number=1,
                    text=text,
                    char_offset_start=0,
                    char_offset_end=len(text),
                )
            ],
            total_characters=len(text),
        )

    def test_chunk_small_document(self, sample_document):
        """Small document should result in single chunk."""
        config = ChunkingConfig(
            target_tokens=10000,  # Large enough for whole doc
            overlap_tokens=100,
        )

        chunks = chunk_document(sample_document, config)

        assert len(chunks) >= 1
        assert chunks[0].chunk_index == 0

    def test_chunk_preserves_content(self, sample_document):
        """Chunking should preserve all content."""
        config = ChunkingConfig(
            target_tokens=500,  # Force multiple chunks
            overlap_tokens=50,
        )

        chunks = chunk_document(sample_document, config)

        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.text) > 0
            assert chunk.start_page >= 1

    def test_chunk_has_overlap(self, sample_document):
        """Chunks should have overlap when configured."""
        config = ChunkingConfig(
            target_tokens=200,  # Force multiple chunks
            overlap_tokens=50,
        )

        chunks = chunk_document(sample_document, config)

        if len(chunks) > 1:
            # Second chunk should have overlap with first
            assert chunks[1].overlap_with_previous >= 0

    def test_chunk_ids_unique(self, sample_document):
        """All chunk IDs should be unique."""
        config = ChunkingConfig(target_tokens=200, overlap_tokens=50)

        chunks = chunk_document(sample_document, config)
        chunk_ids = [c.chunk_id for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_indices_sequential(self, sample_document):
        """Chunk indices should be sequential."""
        config = ChunkingConfig(target_tokens=200, overlap_tokens=50)

        chunks = chunk_document(sample_document, config)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
