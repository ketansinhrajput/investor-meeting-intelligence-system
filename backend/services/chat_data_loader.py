"""Cached data loading and summary generation for the chatbot.

Loads run data into memory and caches it with LRU to avoid
re-reading JSON files on every chat message.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).parent.parent / "data" / "runs"


@lru_cache(maxsize=10)
def load_run_data(run_id: str) -> dict:
    """Load all run data into memory. Cached per run_id (up to 10 runs).

    Returns dict with keys: run_id, metadata, meta, speakers, qa, strategic, extraction
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_id}")

    data = {"run_id": run_id}
    file_map = {
        "metadata": "metadata.json",
        "meta": "stage_metadata_result.json",
        "speakers": "stage_speakers_result.json",
        "qa": "stage_qa_result.json",
        "strategic": "stage_strategic_result.json",
        "extraction": "stage_extraction_result.json",
    }

    for key, filename in file_map.items():
        filepath = run_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
        else:
            data[key] = {}

    logger.info(f"Loaded run data for {run_id}: "
                f"{len(data.get('qa', {}).get('qa_units', []))} Q&As, "
                f"{len(data.get('speakers', {}).get('speakers', {}))} speakers")
    return data


def generate_data_summary(run_data: dict) -> dict:
    """Generate a structured summary of the run data.

    Used for system prompt injection and suggested questions.
    Returns dict with company, quarter, year, speaker info, qa count, etc.
    """
    meta = run_data.get("meta", {})
    speakers_data = run_data.get("speakers", {})
    qa_data = run_data.get("qa", {})
    strategic_data = run_data.get("strategic", {})
    metadata = run_data.get("metadata", {})

    speakers = speakers_data.get("speakers", {})
    management = [s["canonical_name"] for s in speakers.values() if s.get("role") == "management"]
    analysts = [s["canonical_name"] for s in speakers.values() if s.get("role") == "analyst"]
    moderators = [s["canonical_name"] for s in speakers.values() if s.get("role") == "moderator"]

    qa_units = qa_data.get("qa_units", [])
    follow_ups = [q for q in qa_units if q.get("is_follow_up")]

    # Unique questioners
    questioners = list({q.get("questioner_name", "Unknown") for q in qa_units})

    return {
        "company": meta.get("company_name", "Unknown Company"),
        "ticker": meta.get("ticker_symbol", "?"),
        "quarter": meta.get("fiscal_quarter", "?"),
        "year": meta.get("fiscal_year", "?"),
        "call_date": meta.get("call_date", "?"),
        "speaker_count": len(speakers),
        "management_names": management,
        "analyst_names": analysts,
        "moderator_names": moderators,
        "qa_count": len(qa_units),
        "follow_up_count": len(follow_ups),
        "strategic_count": strategic_data.get("total_statements", 0),
        "page_count": metadata.get("page_count", "?"),
        "unique_questioners": questioners,
        "status": metadata.get("status", "unknown"),
    }


def get_summary_text(summary: dict) -> str:
    """Format summary dict as a human-readable string for prompts."""
    mgmt = ", ".join(summary["management_names"]) if summary["management_names"] else "None identified"
    analysts = ", ".join(summary["analyst_names"]) if summary["analyst_names"] else "None identified"

    return (
        f"{summary['company']} ({summary['ticker']}) "
        f"{summary['quarter']} {summary['year']} earnings call. "
        f"{summary['speaker_count']} speakers "
        f"({len(summary['management_names'])} management, {len(summary['analyst_names'])} analysts), "
        f"{summary['qa_count']} Q&A exchanges, "
        f"{summary['page_count']} pages."
    )


def invalidate_cache(run_id: Optional[str] = None):
    """Clear the data cache. If run_id given, only clears that entry."""
    # lru_cache doesn't support selective invalidation,
    # so we clear the entire cache
    load_run_data.cache_clear()
    logger.info(f"Cache cleared" + (f" (triggered by {run_id})" if run_id else ""))
