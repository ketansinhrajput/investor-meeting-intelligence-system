"""Unit tests for Pydantic models."""

import pytest
from datetime import datetime

from src.models import (
    CallMetadata,
    CallPhaseType,
    DocumentChunk,
    EnrichedQAUnit,
    EvidenceReference,
    InferredTopic,
    InvestorIntent,
    InvestorIntentType,
    PageContent,
    ProcessingError,
    ErrorSeverity,
    QAUnit,
    RawDocument,
    ResponsePosture,
    ResponsePostureType,
    SpeakerProfile,
    SpeakerRegistry,
    SpeakerRole,
    SpeakerTurn,
    StructuredReport,
)


class TestPageContent:
    """Tests for PageContent model."""

    def test_valid_page_content(self):
        page = PageContent(
            page_number=1,
            text="Sample text content",
            char_offset_start=0,
            char_offset_end=19,
        )
        assert page.page_number == 1
        assert page.text == "Sample text content"

    def test_invalid_page_number(self):
        with pytest.raises(ValueError):
            PageContent(
                page_number=0,  # Must be >= 1
                text="Text",
                char_offset_start=0,
                char_offset_end=4,
            )


class TestRawDocument:
    """Tests for RawDocument model."""

    def test_valid_document(self):
        doc = RawDocument(
            source_file="test.pdf",
            total_pages=2,
            pages=[
                PageContent(page_number=1, text="Page 1", char_offset_start=0, char_offset_end=6),
                PageContent(page_number=2, text="Page 2", char_offset_start=7, char_offset_end=13),
            ],
            total_characters=13,
        )
        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    def test_full_text_property(self):
        doc = RawDocument(
            source_file="test.pdf",
            total_pages=2,
            pages=[
                PageContent(page_number=1, text="Hello", char_offset_start=0, char_offset_end=5),
                PageContent(page_number=2, text="World", char_offset_start=6, char_offset_end=11),
            ],
            total_characters=11,
        )
        assert doc.full_text == "Hello\nWorld"


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_valid_chunk(self):
        chunk = DocumentChunk(
            chunk_id="chunk_abc123",
            chunk_index=0,
            text="Chunk content",
            start_page=1,
            end_page=2,
            char_offset_start=0,
            char_offset_end=100,
            token_count=25,
        )
        assert chunk.chunk_id == "chunk_abc123"
        assert chunk.token_count == 25


class TestSpeakerModels:
    """Tests for speaker-related models."""

    def test_speaker_role_enum(self):
        assert SpeakerRole.MANAGEMENT.value == "management"
        assert SpeakerRole.INVESTOR_ANALYST.value == "investor_analyst"

    def test_speaker_turn(self):
        turn = SpeakerTurn(
            turn_id="turn_001",
            speaker_name="John Smith",
            speaker_id="john_smith",
            inferred_role=SpeakerRole.MANAGEMENT,
            text="Thank you for the question.",
            start_char=0,
            end_char=28,
            page_number=5,
        )
        assert turn.inferred_role == SpeakerRole.MANAGEMENT

    def test_speaker_registry(self):
        registry = SpeakerRegistry()
        profile = SpeakerProfile(
            speaker_id="john_smith",
            canonical_name="John Smith",
            role=SpeakerRole.MANAGEMENT,
            title="CEO",
            mention_count=5,
        )
        registry.add_speaker(profile)

        assert registry.get_speaker("john_smith") is not None
        assert registry.get_speaker("john_smith").title == "CEO"


class TestQAModels:
    """Tests for Q&A models."""

    def test_qa_unit(self):
        unit = QAUnit(
            unit_id="qa_001",
            sequence_number=1,
            question_turns=["turn_1"],
            questioner_id="analyst_1",
            questioner_name="Michael Chen",
            response_turns=["turn_2", "turn_3"],
            responders=["ceo", "cfo"],
            start_page=5,
            end_page=6,
        )
        assert len(unit.response_turns) == 2

    def test_enriched_qa_unit(self):
        unit = EnrichedQAUnit(
            unit_id="qa_001",
            sequence_number=1,
            question_text="What is your guidance?",
            response_text="We expect strong growth.",
            topics=[
                InferredTopic(
                    topic_name="Guidance",
                    topic_category="Financial",
                    evidence_spans=["We expect strong growth"],
                )
            ],
            investor_intent=InvestorIntent(
                primary_intent=InvestorIntentType.CLARIFICATION,
                reasoning="Asking for specific numbers",
            ),
            response_posture=ResponsePosture(
                primary_posture=ResponsePostureType.CONFIDENT,
                reasoning="Direct answer with specifics",
            ),
            questioner_id="analyst_1",
            responders=["ceo"],
            start_page=5,
            end_page=5,
        )
        assert unit.investor_intent.primary_intent == InvestorIntentType.CLARIFICATION


class TestReportModels:
    """Tests for report models."""

    def test_call_metadata(self):
        metadata = CallMetadata(
            company_name="ACME Corp",
            ticker_symbol="ACME",
            fiscal_quarter="Q3",
            fiscal_year=2024,
            source_file="acme_q3_2024.pdf",
            extraction_timestamp=datetime.utcnow(),
            total_pages=15,
        )
        assert metadata.company_name == "ACME Corp"

    def test_call_metadata_nullable_fields(self):
        metadata = CallMetadata(
            source_file="unknown.pdf",
            extraction_timestamp=datetime.utcnow(),
            total_pages=10,
        )
        assert metadata.company_name is None
        assert metadata.ticker_symbol is None

    def test_processing_error(self):
        error = ProcessingError(
            error_id="err_001",
            severity=ErrorSeverity.WARNING,
            stage="segmentation",
            message="Could not identify speaker role",
            recoverable=True,
        )
        assert error.severity == ErrorSeverity.WARNING


class TestEnums:
    """Tests for enum values."""

    def test_call_phase_types(self):
        assert CallPhaseType.OPENING_REMARKS.value == "opening_remarks"
        assert CallPhaseType.QA_SESSION.value == "qa_session"

    def test_investor_intent_types(self):
        assert InvestorIntentType.CONCERN.value == "concern"
        assert InvestorIntentType.VALIDATION.value == "validation"

    def test_response_posture_types(self):
        assert ResponsePostureType.CONFIDENT.value == "confident"
        assert ResponsePostureType.EVASIVE.value == "evasive"
