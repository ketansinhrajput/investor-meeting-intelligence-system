"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_transcript_text() -> str:
    """Sample earnings call transcript text for testing."""
    return """
ACME Corporation Q3 2024 Earnings Call
October 15, 2024

Operator: Good morning and welcome to the ACME Corporation third quarter 2024 earnings conference call. At this time, all participants are in a listen-only mode. Following management's prepared remarks, we will hold a question-and-answer session.

I would now like to turn the call over to Jane Smith, Vice President of Investor Relations. Please go ahead.

Jane Smith: Thank you, operator. Good morning, everyone, and thank you for joining us today. With me on the call are John Johnson, our Chief Executive Officer, and Sarah Williams, our Chief Financial Officer.

Before we begin, I'd like to remind you that today's discussion will include forward-looking statements. Actual results may differ materially from those projected.

I'll now turn the call over to John.

John Johnson: Thank you, Jane, and good morning, everyone. I'm pleased to report that Q3 was another strong quarter for ACME. We delivered revenue of $2.5 billion, up 12% year-over-year, and adjusted EPS of $1.45, exceeding our guidance.

Our growth was driven by continued strength in our core product lines and successful expansion into new markets. We're particularly excited about the momentum we're seeing in the enterprise segment, which grew 25% this quarter.

Looking ahead, we remain confident in our full-year outlook and are raising our revenue guidance to $10 billion.

I'll now turn the call over to Sarah to discuss our financial results in more detail.

Sarah Williams: Thank you, John. As John mentioned, we had a strong third quarter. Revenue came in at $2.5 billion, representing 12% growth year-over-year.

Gross margin was 65%, up 200 basis points from the prior year, driven by operational efficiencies and favorable product mix.

Operating expenses were $800 million, or 32% of revenue, reflecting continued investment in R&D and sales capacity.

We generated $500 million in free cash flow during the quarter and returned $200 million to shareholders through dividends and share repurchases.

For Q4, we expect revenue of $2.6 to $2.7 billion and adjusted EPS of $1.50 to $1.55.

With that, I'll turn it back to the operator for Q&A.

Operator: Thank you. We will now begin the question-and-answer session. Our first question comes from Michael Chen with Goldman Sachs.

Michael Chen: Hi, good morning. Thanks for taking my question. John, can you provide more color on the enterprise segment growth? What's driving the 25% increase, and how sustainable do you think this growth rate is?

John Johnson: Great question, Michael. The enterprise growth is being driven by a few factors. First, we've seen strong adoption of our new cloud platform among large enterprises. Second, our expanded sales team is gaining traction in key verticals like financial services and healthcare.

In terms of sustainability, while I won't guide to a specific growth rate, we believe the enterprise opportunity remains significant. Our pipeline is strong, and we're investing to capture this market.

Michael Chen: That's helpful. And just a follow-up on the margin expansion - Sarah, how should we think about gross margins going forward? Is 65% a good run rate?

Sarah Williams: Thanks, Michael. We're pleased with the margin performance this quarter. The improvement came from both cost efficiencies and a shift toward higher-margin products. Going forward, we expect gross margins to remain in the 64-66% range, with some quarterly variability depending on product mix.

Operator: Our next question comes from Lisa Park with Morgan Stanley.

Lisa Park: Good morning. I wanted to ask about the competitive environment. We've seen some new entrants in your core market. How are you thinking about pricing and market share?

John Johnson: Thanks, Lisa. Competition is something we monitor closely. While there are new entrants, we believe our technology leadership and customer relationships provide strong differentiation. We're not seeing significant pricing pressure at this point, and our win rates remain stable.

That said, we're not complacent. We continue to invest in innovation and customer success to maintain our competitive position.

Lisa Park: Got it. And one more question on capital allocation - with the strong cash flow, are you considering any M&A opportunities?

Sarah Williams: Lisa, we're always evaluating opportunities to deploy capital strategically. M&A is certainly part of our toolkit, but we're disciplined in our approach. We'll pursue acquisitions that align with our strategy and create shareholder value. In the meantime, we remain committed to returning capital through dividends and buybacks.

Operator: Thank you. This concludes our question-and-answer session. I'll turn the call back to John for closing remarks.

John Johnson: Thank you, operator, and thank you all for joining us today. We're proud of our Q3 results and excited about the opportunities ahead. We remain focused on executing our strategy and delivering value for our shareholders.

Thank you, and have a great day.

Operator: This concludes today's conference call. You may now disconnect.
"""


@pytest.fixture
def sample_raw_document(sample_transcript_text: str) -> dict:
    """Sample RawDocument dict for testing."""
    from datetime import datetime

    return {
        "source_file": "test_transcript.pdf",
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "total_pages": 3,
        "pages": [
            {
                "page_number": 1,
                "text": sample_transcript_text[:2000],
                "char_offset_start": 0,
                "char_offset_end": 2000,
            },
            {
                "page_number": 2,
                "text": sample_transcript_text[2000:4000],
                "char_offset_start": 2000,
                "char_offset_end": 4000,
            },
            {
                "page_number": 3,
                "text": sample_transcript_text[4000:],
                "char_offset_start": 4000,
                "char_offset_end": len(sample_transcript_text),
            },
        ],
        "total_characters": len(sample_transcript_text),
    }
