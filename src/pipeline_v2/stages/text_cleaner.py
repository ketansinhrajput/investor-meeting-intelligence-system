"""Stage 1B: Text Cleaner - Regex-based noise removal.

HYBRID APPROACH:
- Regex removes OBVIOUS noise patterns
- LLM never sees this noise
- Nothing semantic is removed

PATTERNS REMOVED:
1. Company name + date headers (repeated on each page)
2. "Page X of Y" markers
3. Repeated earnings call titles
4. Excessive whitespace
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TextCleaningTrace:
    """Trace of what was cleaned and why."""
    page_headers_removed: list[str] = field(default_factory=list)
    page_markers_removed: list[str] = field(default_factory=list)
    title_headers_removed: list[str] = field(default_factory=list)
    lines_before: int = 0
    lines_after: int = 0
    chars_removed: int = 0


@dataclass
class CleanedText:
    """Result of text cleaning."""
    text: str
    trace: TextCleaningTrace


# =============================================================================
# Noise Patterns
# =============================================================================

# Page markers: "Page 1 of 16", "Page 2 of 16", etc.
PAGE_MARKER_PATTERN = re.compile(
    r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Common earnings call title patterns
TITLE_PATTERNS = [
    # "GMM Pfaudler Limited Q1 FY-22 Earnings Conference Call"
    re.compile(
        r"^.*(?:Earnings|Conference|Quarterly|Annual)\s+(?:Call|Conference|Results).*$",
        re.IGNORECASE | re.MULTILINE
    ),
]

# Date-only lines: "August 12, 2021" or "November 06, 2025"
DATE_LINE_PATTERN = re.compile(
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Company name + date headers (detected dynamically)
# Pattern: "GMM Pfaudler Limited August 12, 2021" or "Company Name\nDate"


def _detect_page_header_pattern(text: str) -> Optional[re.Pattern]:
    """Detect the repeating page header pattern.

    Looks for company name + date that repeats across pages.
    Returns a compiled pattern if found.
    """
    lines = text.split('\n')

    # Look for patterns that appear multiple times
    # Typically: "Company Name" followed by date, or "Company Name Date"

    # Find lines that look like company headers
    header_candidates = []
    for i, line in enumerate(lines[:100]):  # Check first 100 lines
        line = line.strip()
        if not line:
            continue

        # Check if this line + next line form a header
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Company name followed by date
            if (len(line) > 10 and
                re.search(r'Limited|Inc\.|Corp\.|LLC|Pvt\.|Ltd\.', line, re.IGNORECASE) and
                DATE_LINE_PATTERN.match(next_line)):
                header_candidates.append((line, next_line))

    if not header_candidates:
        return None

    # Find most common header
    from collections import Counter
    header_counts = Counter(header_candidates)
    most_common = header_counts.most_common(1)

    if most_common and most_common[0][1] >= 2:  # Must appear at least twice
        company, date = most_common[0][0]
        # Create pattern to match this header
        company_escaped = re.escape(company)
        pattern = re.compile(
            rf"^\s*{company_escaped}\s*$\n^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{{1,2}},?\s+\d{{4}}\s*$",
            re.IGNORECASE | re.MULTILINE
        )
        return pattern

    return None


def _detect_inline_header_pattern(text: str, company_name: Optional[str] = None) -> Optional[re.Pattern]:
    """Detect inline headers like 'GMM Pfaudler Limited August 12, 2021'.

    These often appear at the top of each page.
    """
    if not company_name:
        # Try to extract company name from first few lines
        lines = text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if re.search(r'Limited|Inc\.|Corp\.|LLC|Pvt\.|Ltd\.', line, re.IGNORECASE):
                # Extract company name (before date if inline)
                match = re.match(
                    r'^(.*?(?:Limited|Inc\.|Corp\.|LLC|Pvt\.|Ltd\.))',
                    line,
                    re.IGNORECASE
                )
                if match:
                    company_name = match.group(1).strip()
                    break

    if not company_name:
        return None

    # Create pattern for inline header: "Company Name Month DD, YYYY"
    company_escaped = re.escape(company_name)
    pattern = re.compile(
        rf"^\s*{company_escaped}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{{1,2}},?\s+\d{{4}}\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    return pattern


# =============================================================================
# Main Cleaning Function
# =============================================================================

def clean_extracted_text(
    raw_text: str,
    company_name: Optional[str] = None,
) -> CleanedText:
    """Clean extracted text by removing noise patterns.

    RULES:
    - Remove page markers ("Page X of Y")
    - Remove repeated page headers (Company + Date)
    - Remove repeated earnings call titles
    - Normalize excessive whitespace
    - NEVER remove semantic content

    Args:
        raw_text: Raw extracted text from PDF
        company_name: Optional company name for header detection

    Returns:
        CleanedText with cleaned text and trace
    """
    trace = TextCleaningTrace()
    trace.lines_before = len(raw_text.split('\n'))
    original_len = len(raw_text)

    text = raw_text

    # 1. Remove page markers
    page_markers = PAGE_MARKER_PATTERN.findall(text)
    trace.page_markers_removed = [m.strip() for m in page_markers]
    text = PAGE_MARKER_PATTERN.sub('', text)

    logger.debug("page_markers_removed", count=len(page_markers))

    # 2. Detect and remove page headers (Company + Date)
    header_pattern = _detect_page_header_pattern(text)
    if header_pattern:
        headers = header_pattern.findall(text)
        trace.page_headers_removed = [h.strip() if isinstance(h, str) else str(h) for h in headers]
        text = header_pattern.sub('', text)
        logger.debug("page_headers_removed", count=len(headers))

    # 3. Detect and remove inline headers
    inline_pattern = _detect_inline_header_pattern(text, company_name)
    if inline_pattern:
        inline_headers = inline_pattern.findall(text)
        trace.page_headers_removed.extend([h.strip() for h in inline_headers])
        text = inline_pattern.sub('', text)
        logger.debug("inline_headers_removed", count=len(inline_headers))

    # 4. Remove standalone date lines that are likely headers
    # Only if they appear multiple times (indicating page headers)
    date_lines = DATE_LINE_PATTERN.findall(text)
    if len(date_lines) >= 3:  # Multiple occurrences suggest page headers
        text = DATE_LINE_PATTERN.sub('', text)
        trace.page_headers_removed.extend([d.strip() for d in date_lines])
        logger.debug("date_headers_removed", count=len(date_lines))

    # 5. Normalize whitespace
    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final cleanup
    text = text.strip()

    trace.lines_after = len(text.split('\n'))
    trace.chars_removed = original_len - len(text)

    logger.info(
        "text_cleaning_complete",
        lines_before=trace.lines_before,
        lines_after=trace.lines_after,
        chars_removed=trace.chars_removed,
        page_markers=len(trace.page_markers_removed),
        headers=len(trace.page_headers_removed),
    )

    return CleanedText(text=text, trace=trace)
