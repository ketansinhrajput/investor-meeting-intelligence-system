"""Enumeration types for the pipeline models."""

from enum import Enum


class SpeakerRole(str, Enum):
    """Role classification for call participants."""

    MODERATOR = "moderator"
    INVESTOR_ANALYST = "investor_analyst"
    MANAGEMENT = "management"
    UNKNOWN = "unknown"


class CallPhaseType(str, Enum):
    """Phases of an earnings call."""

    OPENING_REMARKS = "opening_remarks"
    QA_SESSION = "qa_session"
    CLOSING_REMARKS = "closing_remarks"
    TRANSITION = "transition"


class InvestorIntentType(str, Enum):
    """Classification of investor question intent."""

    CONCERN = "concern"
    CLARIFICATION = "clarification"
    VALIDATION = "validation"
    EXPLORATION = "exploration"
    CHALLENGE = "challenge"
    FOLLOW_UP = "follow_up"


class ResponsePostureType(str, Enum):
    """Classification of management response posture."""

    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    EVASIVE = "evasive"
    TRANSPARENT = "transparent"
    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"


class StatementType(str, Enum):
    """Classification of strategic statement types."""

    GUIDANCE = "guidance"
    OUTLOOK = "outlook"
    STRATEGIC_INITIATIVE = "strategic_initiative"
    OPERATIONAL_UPDATE = "operational_update"
    FINANCIAL_HIGHLIGHT = "financial_highlight"
    RISK_DISCLOSURE = "risk_disclosure"
    OTHER = "other"


class SentimentType(str, Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ErrorSeverity(str, Enum):
    """Severity levels for processing errors."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
