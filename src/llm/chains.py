"""LangChain chains for LLM operations."""

import json
import re

import structlog
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_USER_PROMPT,
    METADATA_EXTRACTION_PROMPT,
    QA_EXTRACTION_SYSTEM_PROMPT,
    QA_EXTRACTION_USER_PROMPT,
    SEGMENTATION_SYSTEM_PROMPT,
    SEGMENTATION_USER_PROMPT,
    STRATEGIC_EXTRACTION_SYSTEM_PROMPT,
    STRATEGIC_EXTRACTION_USER_PROMPT,
)
from src.llm.client import create_json_llm_client, get_fallback_model_name, get_primary_model_name

logger = structlog.get_logger(__name__)


class LLMChainError(Exception):
    """Error during LLM chain execution."""

    pass


def _extract_json_from_text(text: str) -> str | None:
    """Try to extract JSON object from text that may contain other content.

    Handles cases where model outputs thinking/reasoning before JSON,
    or wraps JSON in code blocks or quotes.

    Args:
        text: Text that may contain JSON.

    Returns:
        Extracted JSON string or None.
    """
    # Try to find JSON object by matching braces
    # Start from the first { to skip any preamble/thinking
    brace_count = 0
    start_idx = None
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        # Handle string escaping to avoid counting braces inside strings
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                return text[start_idx:i + 1]

    return None


def _clean_json_string(text: str) -> str:
    """Clean common issues in JSON strings from LLM output.

    Args:
        text: Raw JSON string.

    Returns:
        Cleaned JSON string.
    """
    # Remove any BOM or zero-width characters
    text = text.strip('\ufeff\u200b\u200c\u200d')

    # Fix common escape issues
    # Sometimes models output literal \n instead of actual newlines in strings
    # which is actually correct JSON, so we don't change that

    # Remove trailing commas before } or ] (invalid JSON but common LLM mistake)
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    return text


def _invoke_with_fallback(
    prompt: ChatPromptTemplate,
    variables: dict,
    context_name: str = "chain",
) -> tuple[str, str]:
    """Invoke LLM chain with automatic fallback on empty response.

    Args:
        prompt: The ChatPromptTemplate to use.
        variables: Variables to pass to the prompt.
        context_name: Name for logging context.

    Returns:
        Tuple of (response_text, model_used).

    Raises:
        LLMChainError: If both primary and fallback return empty.
    """
    # Try primary model first
    primary_model = get_primary_model_name()
    llm = create_json_llm_client(use_fallback=False)
    chain = prompt | llm | StrOutputParser()

    logger.debug(f"{context_name}_trying_primary", model=primary_model)
    response = chain.invoke(variables)

    if response and response.strip():
        logger.debug(f"{context_name}_primary_success", model=primary_model, length=len(response))
        return response, primary_model

    # Primary returned empty, try fallback
    fallback_model = get_fallback_model_name()
    logger.warning(
        f"{context_name}_primary_empty_trying_fallback",
        primary_model=primary_model,
        fallback_model=fallback_model,
    )

    llm_fallback = create_json_llm_client(use_fallback=True)
    chain_fallback = prompt | llm_fallback | StrOutputParser()
    response = chain_fallback.invoke(variables)

    if response and response.strip():
        logger.info(f"{context_name}_fallback_success", model=fallback_model, length=len(response))
        return response, fallback_model

    # Both failed
    raise LLMChainError(f"Both primary ({primary_model}) and fallback ({fallback_model}) returned empty responses")


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response, handling common issues.

    Handles cases where model outputs thinking/reasoning before JSON,
    wraps JSON in code blocks, or has other formatting issues.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed JSON dict.

    Raises:
        LLMChainError: If JSON parsing fails.
    """
    if not response or not response.strip():
        raise LLMChainError("Empty response from LLM")

    original_response = response
    text = response.strip()

    # Log the raw response for debugging
    logger.debug(
        "raw_llm_response",
        response_length=len(text),
        preview=text[:500] if len(text) > 500 else text,
        ends_with=text[-100:] if len(text) > 100 else text,
    )

    # Strategy 1: Remove markdown code blocks if present
    # Handle various code block formats: ```json, ```JSON, ``` json, etc.
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```JSON\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_block = match.group(1).strip()
            if extracted_block.startswith('{'):
                text = extracted_block
                break

    # Strategy 2: If text doesn't start with {, try to find JSON after thinking text
    if not text.startswith('{'):
        # Look for common patterns that precede JSON
        # e.g., "Here's the JSON:", "Output:", "Result:", etc.
        split_patterns = [
            r'(?:here\'?s?\s+(?:the\s+)?(?:json|output|result)\s*:?\s*)',
            r'(?:output\s*:?\s*)',
            r'(?:result\s*:?\s*)',
            r'(?:response\s*:?\s*)',
        ]
        for pattern in split_patterns:
            parts = re.split(pattern, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                remainder = parts[-1].strip()
                if remainder.startswith('{'):
                    text = remainder
                    break

    text = text.strip()

    # Strategy 3: Direct parsing attempt
    try:
        cleaned = _clean_json_string(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug("direct_parse_failed", error=str(e))

    # Strategy 4: Extract JSON by brace matching from cleaned text
    extracted = _extract_json_from_text(text)
    if extracted:
        try:
            cleaned = _clean_json_string(extracted)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug("extracted_parse_failed", error=str(e))

    # Strategy 5: Try from original response (in case cleaning removed something important)
    extracted = _extract_json_from_text(original_response)
    if extracted:
        try:
            cleaned = _clean_json_string(extracted)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug("original_extracted_parse_failed", error=str(e))

    # Strategy 6: Last resort - try to find ANY valid JSON object
    # Sometimes models output multiple JSON attempts, find the largest valid one
    json_candidates = []
    for match in re.finditer(r'\{[^{}]*\}|\{[^{}]*\{[^{}]*\}[^{}]*\}', text):
        try:
            candidate = json.loads(match.group())
            json_candidates.append((len(match.group()), candidate))
        except json.JSONDecodeError:
            continue

    if json_candidates:
        # Return the largest valid JSON object found
        json_candidates.sort(reverse=True)
        return json_candidates[0][1]

    logger.error(
        "json_parse_error",
        error="Could not extract valid JSON",
        response_preview=text[:300] if len(text) > 300 else text
    )
    raise LLMChainError(f"Failed to parse LLM JSON response. Response preview: {text[:150]}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def run_segmentation_chain(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    start_page: int,
    previous_speaker: str | None = None,
    previous_phase: str | None = None,
) -> dict:
    """Run the segmentation chain on a chunk.

    Args:
        chunk_text: Text of the chunk to segment.
        chunk_index: Index of this chunk.
        total_chunks: Total number of chunks.
        start_page: Starting page number.
        previous_speaker: Speaker from previous chunk (for continuity).
        previous_phase: Phase from previous chunk.

    Returns:
        Segmentation result dict with turns and phases.
    """
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SEGMENTATION_SYSTEM_PROMPT),
        ("human", SEGMENTATION_USER_PROMPT),
    ])

    logger.debug(
        "running_segmentation",
        chunk_index=chunk_index,
        text_length=len(chunk_text),
    )

    # Use fallback-enabled invocation
    response, model_used = _invoke_with_fallback(
        prompt=prompt,
        variables={
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "start_page": start_page,
            "previous_speaker": previous_speaker or "None",
            "previous_phase": previous_phase or "unknown",
        },
        context_name="segmentation",
    )

    # Debug log the raw response before parsing
    logger.debug(
        "segmentation_raw_response",
        chunk_index=chunk_index,
        model_used=model_used,
        response_length=len(response) if response else 0,
        response_preview=response[:200] if response else "EMPTY",
    )

    # Try JsonOutputParser first, fallback to manual parsing
    try:
        result = json_parser.parse(response)
    except Exception as e:
        logger.debug("json_parser_failed", error=str(e))
        result = _parse_json_response(response)

    logger.debug(
        "segmentation_complete",
        chunk_index=chunk_index,
        model_used=model_used,
        turns_found=len(result.get("turns", [])),
    )

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def run_qa_extraction_chain(turns_json: str) -> dict:
    """Run Q&A extraction chain on segmented turns.

    Args:
        turns_json: JSON string of speaker turns.

    Returns:
        Q&A extraction result with qa_units array.
    """
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_EXTRACTION_SYSTEM_PROMPT),
        ("human", QA_EXTRACTION_USER_PROMPT),
    ])

    logger.debug("running_qa_extraction")

    response, model_used = _invoke_with_fallback(
        prompt=prompt,
        variables={"turns_json": turns_json},
        context_name="qa_extraction",
    )

    try:
        result = json_parser.parse(response)
    except Exception:
        result = _parse_json_response(response)

    logger.debug(
        "qa_extraction_complete",
        model_used=model_used,
        units_found=len(result.get("qa_units", [])),
    )

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def run_strategic_extraction_chain(turns_json: str) -> dict:
    """Run strategic statement extraction chain.

    Args:
        turns_json: JSON string of speaker turns from opening/closing phases.

    Returns:
        Strategic extraction result with strategic_statements array.
    """
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", STRATEGIC_EXTRACTION_SYSTEM_PROMPT),
        ("human", STRATEGIC_EXTRACTION_USER_PROMPT),
    ])

    logger.debug("running_strategic_extraction")

    response, model_used = _invoke_with_fallback(
        prompt=prompt,
        variables={"turns_json": turns_json},
        context_name="strategic_extraction",
    )

    try:
        result = json_parser.parse(response)
    except Exception:
        result = _parse_json_response(response)

    logger.debug(
        "strategic_extraction_complete",
        model_used=model_used,
        statements_found=len(result.get("strategic_statements", [])),
    )

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def run_enrichment_chain(
    unit_type: str,
    content: str,
    start_page: int,
    end_page: int,
) -> dict:
    """Run enrichment chain on a Q&A unit or strategic statement.

    Args:
        unit_type: "Q&A Unit" or "Strategic Statement".
        content: The content to enrich.
        start_page: Starting page number.
        end_page: Ending page number.

    Returns:
        Enrichment result with topics, intent, posture, evidence.
    """
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", ENRICHMENT_SYSTEM_PROMPT),
        ("human", ENRICHMENT_USER_PROMPT),
    ])

    logger.debug("running_enrichment", unit_type=unit_type)

    response, model_used = _invoke_with_fallback(
        prompt=prompt,
        variables={
            "unit_type": unit_type,
            "content": content,
            "start_page": start_page,
            "end_page": end_page,
        },
        context_name="enrichment",
    )

    try:
        result = json_parser.parse(response)
    except Exception:
        result = _parse_json_response(response)

    logger.debug(
        "enrichment_complete",
        unit_type=unit_type,
        model_used=model_used,
        topics_found=len(result.get("topics", [])),
    )

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def run_metadata_extraction_chain(text: str) -> dict:
    """Extract call metadata from transcript header.

    Args:
        text: First 1-2 pages of transcript text.

    Returns:
        Metadata dict with company_name, ticker, quarter, year, date.
    """
    json_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("human", METADATA_EXTRACTION_PROMPT),
    ])

    logger.debug("extracting_metadata")

    response, model_used = _invoke_with_fallback(
        prompt=prompt,
        variables={"text": text},
        context_name="metadata_extraction",
    )

    try:
        result = json_parser.parse(response)
    except Exception:
        result = _parse_json_response(response)

    logger.debug("metadata_extraction_complete", model_used=model_used, result=result)

    return result
