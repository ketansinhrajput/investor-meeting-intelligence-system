"""Chunk aggregation pipeline node."""

import uuid

import structlog
from rapidfuzz import fuzz

from src.models import ErrorSeverity, SpeakerRole
from src.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

# Threshold for fuzzy name matching
NAME_MATCH_THRESHOLD = 85


def aggregate_chunks_node(state: PipelineState) -> PipelineState:
    """Aggregate segmented chunks into unified transcript.

    Handles:
    - Speaker identity resolution across chunks
    - Deduplication of overlapping turns
    - Global turn ordering
    - Phase boundary consolidation

    Args:
        state: Current pipeline state with segmented_chunks.

    Returns:
        Updated state with segmented_transcript and speaker_registry.
    """
    logger.info("aggregation_node_start")

    segmented_chunks = state.get("segmented_chunks", [])

    if not segmented_chunks:
        logger.warning("aggregation_node_skip_no_chunks")
        return state

    try:
        # Collect all turns from all chunks
        all_turns = []
        for chunk in segmented_chunks:
            all_turns.extend(chunk.get("turns", []))

        # Resolve speaker identities
        speaker_registry, turn_id_to_speaker = _resolve_speakers(all_turns)

        # Update turns with resolved speaker IDs
        for turn in all_turns:
            old_id = turn["speaker_id"]
            if old_id in turn_id_to_speaker:
                turn["speaker_id"] = turn_id_to_speaker[old_id]

        # Deduplicate turns from overlapping regions
        unique_turns = _deduplicate_turns(all_turns)

        # Sort by character offset
        unique_turns.sort(key=lambda t: t.get("start_char", 0))

        # Consolidate phases
        all_phases = []
        for chunk in segmented_chunks:
            all_phases.extend(chunk.get("detected_phases", []))

        consolidated_phases = _consolidate_phases(all_phases)

        # Build segmented transcript
        segmented_transcript = {
            "turns": unique_turns,
            "phases": consolidated_phases,
            "speaker_registry": speaker_registry,
        }

        logger.info(
            "aggregation_node_complete",
            total_turns=len(unique_turns),
            unique_speakers=len(speaker_registry.get("speakers", {})),
            phases=len(consolidated_phases),
        )

        return {
            **state,
            "segmented_transcript": segmented_transcript,
            "speaker_registry": speaker_registry,
        }

    except Exception as e:
        logger.exception("aggregation_node_error")

        error = {
            "error_id": f"agg_err_{uuid.uuid4().hex[:8]}",
            "severity": ErrorSeverity.ERROR.value,
            "stage": "aggregation",
            "message": str(e),
            "details": {"exception_type": type(e).__name__},
            "recoverable": True,
        }

        return {
            **state,
            "segmented_transcript": None,
            "errors": state.get("errors", []) + [error],
        }


def _resolve_speakers(turns: list[dict]) -> tuple[dict, dict[str, str]]:
    """Resolve speaker identities across turns.

    Groups speakers by similar names and assigns canonical identities.

    Args:
        turns: List of turn dicts.

    Returns:
        Tuple of (speaker_registry dict, old_id to new_id mapping).
    """
    # Collect unique speaker mentions
    speaker_mentions: dict[str, list[dict]] = {}

    for turn in turns:
        speaker_id = turn.get("speaker_id", "")
        speaker_name = turn.get("speaker_name")
        role = turn.get("inferred_role", "unknown")

        if speaker_id not in speaker_mentions:
            speaker_mentions[speaker_id] = []

        speaker_mentions[speaker_id].append({
            "name": speaker_name,
            "role": role,
        })

    # Cluster similar speakers
    clusters: list[list[str]] = []
    processed = set()

    speaker_ids = list(speaker_mentions.keys())

    for i, sid1 in enumerate(speaker_ids):
        if sid1 in processed:
            continue

        cluster = [sid1]
        processed.add(sid1)

        name1 = speaker_mentions[sid1][0].get("name") or ""

        for sid2 in speaker_ids[i + 1:]:
            if sid2 in processed:
                continue

            name2 = speaker_mentions[sid2][0].get("name") or ""

            # Check name similarity
            if name1 and name2:
                similarity = fuzz.ratio(name1.lower(), name2.lower())
                if similarity >= NAME_MATCH_THRESHOLD:
                    cluster.append(sid2)
                    processed.add(sid2)

        clusters.append(cluster)

    # Build speaker profiles and ID mapping
    speakers = {}
    id_mapping = {}

    for cluster in clusters:
        # Find canonical name (most common or longest)
        names = []
        roles = []

        for sid in cluster:
            for mention in speaker_mentions[sid]:
                if mention.get("name"):
                    names.append(mention["name"])
                roles.append(mention.get("role", "unknown"))

        canonical_name = _get_canonical_name(names)

        # Determine role by voting
        role_counts: dict[str, int] = {}
        for r in roles:
            role_counts[r] = role_counts.get(r, 0) + 1

        dominant_role = max(role_counts.items(), key=lambda x: x[1])[0]

        # Create canonical ID
        canonical_id = _generate_speaker_id(canonical_name)

        # Ensure uniqueness
        base_id = canonical_id
        counter = 1
        while canonical_id in speakers:
            canonical_id = f"{base_id}_{counter}"
            counter += 1

        # Create profile
        profile = {
            "speaker_id": canonical_id,
            "canonical_name": canonical_name or "Unknown Speaker",
            "role": dominant_role,
            "title": None,  # Could be extracted from context
            "organization": None,  # Could be extracted from context
            "mention_count": sum(len(speaker_mentions[sid]) for sid in cluster),
        }

        speakers[canonical_id] = profile

        # Map old IDs to new canonical ID
        for sid in cluster:
            id_mapping[sid] = canonical_id

    return {"speakers": speakers}, id_mapping


def _get_canonical_name(names: list[str]) -> str:
    """Get canonical name from list of name variants.

    Prefers longer, more complete names.

    Args:
        names: List of name variants.

    Returns:
        Canonical name.
    """
    if not names:
        return "Unknown Speaker"

    # Count occurrences
    name_counts: dict[str, int] = {}
    for name in names:
        normalized = name.strip()
        name_counts[normalized] = name_counts.get(normalized, 0) + 1

    # Sort by count (descending), then by length (descending)
    sorted_names = sorted(
        name_counts.items(),
        key=lambda x: (x[1], len(x[0])),
        reverse=True,
    )

    return sorted_names[0][0]


def _generate_speaker_id(speaker_name: str | None) -> str:
    """Generate speaker ID from name."""
    if not speaker_name:
        return f"unknown_{uuid.uuid4().hex[:6]}"

    normalized = speaker_name.lower()
    normalized = "".join(c if c.isalnum() or c.isspace() else "" for c in normalized)
    normalized = "_".join(normalized.split())

    return normalized or f"speaker_{uuid.uuid4().hex[:6]}"


def _deduplicate_turns(turns: list[dict]) -> list[dict]:
    """Remove duplicate turns from overlapping regions.

    Uses text similarity and character offset to detect duplicates.

    Args:
        turns: List of turns potentially with duplicates.

    Returns:
        Deduplicated list of turns.
    """
    if not turns:
        return []

    # Sort by start_char
    sorted_turns = sorted(turns, key=lambda t: t.get("start_char", 0))

    unique = [sorted_turns[0]]

    for turn in sorted_turns[1:]:
        last_turn = unique[-1]

        # Check for overlap
        is_duplicate = False

        # Same speaker and very similar text
        if turn.get("speaker_id") == last_turn.get("speaker_id"):
            similarity = fuzz.ratio(
                turn.get("text", "")[:200],
                last_turn.get("text", "")[:200],
            )
            if similarity >= 90:
                is_duplicate = True

        # Check character offset overlap
        if not is_duplicate:
            last_end = last_turn.get("end_char", 0)
            curr_start = turn.get("start_char", 0)

            # Significant overlap suggests duplicate
            if curr_start < last_end - 100:
                similarity = fuzz.ratio(
                    turn.get("text", "")[:200],
                    last_turn.get("text", "")[:200],
                )
                if similarity >= 80:
                    is_duplicate = True

        if not is_duplicate:
            unique.append(turn)

    return unique


def _consolidate_phases(phases: list[dict]) -> list[dict]:
    """Consolidate phase boundaries from multiple chunks.

    Merges adjacent phases of the same type.

    Args:
        phases: List of phase dicts from all chunks.

    Returns:
        Consolidated list of phases.
    """
    if not phases:
        return []

    # Sort by start page
    sorted_phases = sorted(phases, key=lambda p: p.get("start_page", 0))

    consolidated = [sorted_phases[0]]

    for phase in sorted_phases[1:]:
        last_phase = consolidated[-1]

        # Merge if same type and adjacent
        if phase.get("phase_type") == last_phase.get("phase_type"):
            # Extend the last phase
            last_phase["end_turn_id"] = phase.get("end_turn_id")
            last_phase["end_page"] = phase.get("end_page")
        else:
            consolidated.append(phase)

    return consolidated
