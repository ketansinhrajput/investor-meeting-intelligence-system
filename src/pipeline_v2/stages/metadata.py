"""Stage 1: Metadata Extraction - Extract call metadata from transcript header.

Extracts from first 1-2 pages only:
- Company name and ticker
- Fiscal quarter and year
- Call date
- Mentioned participants (for speaker registry seeding)
"""

import re
from datetime import date, datetime
from typing import Optional

import structlog

from src.llm.chains import run_metadata_extraction_chain
from src.pipeline_v2.models import ExtractedMetadata

logger = structlog.get_logger(__name__)


# =============================================================================
# Deterministic Extraction Patterns
# =============================================================================

# Fiscal quarter patterns
QUARTER_PATTERNS = [
    r"(?i)(?:Q|Quarter)\s*([1-4])\s*(?:FY|Fiscal\s*Year)?\s*['\"]?(\d{2,4})",
    r"(?i)(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:FY|fiscal\s+year\s+)?['\"]?(\d{2,4})",
    r"(?i)(?:FY|Fiscal\s*Year)\s*['\"]?(\d{2,4})\s+(?:Q|Quarter)\s*([1-4])",
]

QUARTER_WORD_MAP = {
    "first": "Q1",
    "second": "Q2",
    "third": "Q3",
    "fourth": "Q4",
}

# Date patterns
DATE_PATTERNS = [
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}",
    r"\d{1,2}/\d{1,2}/\d{2,4}",
    r"\d{4}-\d{2}-\d{2}",
]

# Ticker patterns (NYSE/NASDAQ style)
TICKER_PATTERNS = [
    r"\(([A-Z]{1,5})\)",  # (AAPL)
    r"(?:NYSE|NASDAQ|NYSE:\s*|NASDAQ:\s*)([A-Z]{1,5})",  # NYSE: AAPL
    r"(?:ticker|symbol)[:\s]+([A-Z]{1,5})",  # ticker: AAPL
]

# Company name patterns (usually in title/header)
COMPANY_PATTERNS = [
    r"^(.+?)\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|L\.L\.C\.|Company|Co\.)",
    r"(.+?)\s+(?:Earnings|Quarterly|Q[1-4])\s+(?:Call|Conference)",
]

# Participant patterns
PARTICIPANT_PATTERNS = [
    r"(?:CEO|CFO|COO|CTO|President|Chairman|Director|VP|Vice President|Analyst|Managing Director)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–—]\s*(?:CEO|CFO|COO|CTO|President|Chairman|Director|VP|Vice President|Analyst)",
]


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_quarter_deterministic(text: str) -> tuple[Optional[str], Optional[int]]:
    """Extract fiscal quarter and year using patterns.

    Returns:
        Tuple of (quarter_str, year_int) or (None, None).
    """
    # Try Q1-Q4 format first
    for pattern in QUARTER_PATTERNS[:2]:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Could be (quarter, year) or (year, quarter)
                if groups[0].isdigit() and 1 <= int(groups[0]) <= 4:
                    quarter = f"Q{groups[0]}"
                    year_str = groups[1]
                else:
                    # Word form: "first quarter 2024"
                    word = groups[0].lower() if not groups[0].isdigit() else None
                    if word and word in QUARTER_WORD_MAP:
                        quarter = QUARTER_WORD_MAP[word]
                        year_str = groups[1] if groups[1].isdigit() else None
                    else:
                        continue

                if year_str:
                    year = int(year_str)
                    if year < 100:
                        year += 2000
                    return quarter, year

    # Try FY format
    match = re.search(QUARTER_PATTERNS[2], text)
    if match:
        year_str, quarter_num = match.groups()
        year = int(year_str)
        if year < 100:
            year += 2000
        return f"Q{quarter_num}", year

    return None, None


def _extract_date_deterministic(text: str) -> Optional[date]:
    """Extract call date using patterns.

    Returns:
        Extracted date or None.
    """
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            date_str = match.group()
            # Try various date formats
            formats = [
                "%B %d, %Y",
                "%B %d %Y",
                "%d %B, %Y",
                "%d %B %Y",
                "%m/%d/%Y",
                "%m/%d/%y",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
    return None


def _extract_ticker_deterministic(text: str) -> Optional[str]:
    """Extract stock ticker using patterns."""
    for pattern in TICKER_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _extract_company_name_deterministic(text: str) -> Optional[str]:
    """Extract company name using patterns."""
    # Look at first few lines
    first_lines = text[:500].split("\n")

    for line in first_lines[:10]:
        line = line.strip()
        if not line:
            continue

        for pattern in COMPANY_PATTERNS:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                if len(name) > 3:  # Avoid short matches
                    return name

    return None


def _extract_participants_deterministic(text: str) -> list[str]:
    """Extract participant names mentioned in header."""
    participants = []

    for pattern in PARTICIPANT_PATTERNS:
        for match in re.finditer(pattern, text[:3000]):  # First ~2 pages
            name = match.group(1).strip()
            if name and len(name) > 3:
                # Clean up name
                name = re.sub(r"\s+", " ", name)
                if name not in participants:
                    participants.append(name)

    return participants


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_metadata(
    full_text: str,
    header_text: Optional[str] = None,
    use_llm: bool = True,
) -> ExtractedMetadata:
    """Extract call metadata from transcript.

    Args:
        full_text: Complete transcript text.
        header_text: Optional pre-extracted header text (first 1-2 pages).
        use_llm: Whether to use LLM for enhanced extraction.

    Returns:
        ExtractedMetadata with extracted fields.
    """
    logger.info("metadata_extraction_start", text_length=len(full_text), use_llm=use_llm)

    # Use header text or first ~4000 chars (~2 pages)
    text_to_analyze = header_text or full_text[:4000]

    # Step 1: Deterministic extraction
    quarter, year = _extract_quarter_deterministic(text_to_analyze)
    call_date = _extract_date_deterministic(text_to_analyze)
    ticker = _extract_ticker_deterministic(text_to_analyze)
    company = _extract_company_name_deterministic(text_to_analyze)
    participants = _extract_participants_deterministic(text_to_analyze)

    logger.debug(
        "deterministic_extraction",
        quarter=quarter,
        year=year,
        call_date=str(call_date) if call_date else None,
        ticker=ticker,
        company=company,
        participant_count=len(participants),
    )

    # Step 2: LLM extraction to fill gaps
    llm_result = {}
    if use_llm and (not company or not quarter or not year or not call_date):
        try:
            llm_result = run_metadata_extraction_chain(text_to_analyze)
            logger.debug("llm_extraction_result", result=llm_result)
        except Exception as e:
            logger.warning("llm_extraction_failed", error=str(e))

    # Merge results (deterministic takes precedence, LLM fills gaps)
    company = company or llm_result.get("company_name")
    ticker = ticker or llm_result.get("ticker_symbol")

    if not quarter and llm_result.get("fiscal_quarter"):
        quarter = llm_result["fiscal_quarter"]
    if not year and llm_result.get("fiscal_year"):
        year = llm_result["fiscal_year"]

    if not call_date and llm_result.get("call_date"):
        try:
            call_date = datetime.strptime(llm_result["call_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    # Add LLM participants to list
    llm_participants = llm_result.get("mentioned_participants", [])
    for p in llm_participants:
        if p and p not in participants:
            participants.append(p)

    # Build call title if not found
    call_title = llm_result.get("call_title")
    if not call_title and company and quarter and year:
        call_title = f"{company} {quarter} {year} Earnings Call"

    # Compute confidence
    filled_fields = sum([
        bool(company),
        bool(ticker),
        bool(quarter),
        bool(year),
        bool(call_date),
    ])
    confidence = filled_fields / 5.0

    result = ExtractedMetadata(
        company_name=company,
        ticker_symbol=ticker,
        fiscal_quarter=quarter,
        fiscal_year=year,
        call_date=call_date,
        call_title=call_title,
        mentioned_participants=participants,
        source_pages=(1, 2),
        extraction_confidence=confidence,
    )

    logger.info(
        "metadata_extraction_complete",
        company=company,
        ticker=ticker,
        quarter=quarter,
        year=year,
        confidence=confidence,
    )

    return result
