"""Speaker identity resolution utilities."""

from rapidfuzz import fuzz

from src.models import SpeakerProfile, SpeakerRegistry, SpeakerRole

# Threshold for fuzzy name matching
NAME_MATCH_THRESHOLD = 85


def resolve_speakers(
    speaker_mentions: list[dict],
) -> tuple[SpeakerRegistry, dict[str, str]]:
    """Resolve speaker identities from multiple mentions.

    Groups similar speaker names together and creates canonical profiles.

    Args:
        speaker_mentions: List of dicts with 'speaker_id', 'speaker_name', 'role'.

    Returns:
        Tuple of (SpeakerRegistry, old_id_to_new_id_mapping).
    """
    if not speaker_mentions:
        return SpeakerRegistry(), {}

    # Group mentions by original speaker_id
    mentions_by_id: dict[str, list[dict]] = {}
    for mention in speaker_mentions:
        sid = mention.get("speaker_id", "")
        if sid not in mentions_by_id:
            mentions_by_id[sid] = []
        mentions_by_id[sid].append(mention)

    # Cluster similar speakers
    clusters = _cluster_speakers(mentions_by_id)

    # Build profiles and mapping
    registry = SpeakerRegistry()
    id_mapping = {}

    for cluster in clusters:
        profile = _build_profile_from_cluster(cluster, mentions_by_id)
        registry.add_speaker(profile)

        for sid in cluster:
            id_mapping[sid] = profile.speaker_id

    return registry, id_mapping


def _cluster_speakers(mentions_by_id: dict[str, list[dict]]) -> list[list[str]]:
    """Cluster speaker IDs by name similarity.

    Args:
        mentions_by_id: Speaker mentions grouped by ID.

    Returns:
        List of clusters (each cluster is a list of speaker IDs).
    """
    speaker_ids = list(mentions_by_id.keys())
    clusters: list[list[str]] = []
    processed = set()

    for i, sid1 in enumerate(speaker_ids):
        if sid1 in processed:
            continue

        cluster = [sid1]
        processed.add(sid1)

        # Get representative name for sid1
        name1 = _get_representative_name(mentions_by_id[sid1])

        for sid2 in speaker_ids[i + 1:]:
            if sid2 in processed:
                continue

            name2 = _get_representative_name(mentions_by_id[sid2])

            # Check name similarity
            if name1 and name2:
                similarity = fuzz.ratio(name1.lower(), name2.lower())
                if similarity >= NAME_MATCH_THRESHOLD:
                    cluster.append(sid2)
                    processed.add(sid2)

        clusters.append(cluster)

    return clusters


def _get_representative_name(mentions: list[dict]) -> str | None:
    """Get the most representative name from mentions.

    Args:
        mentions: List of mention dicts.

    Returns:
        Most common or longest name, or None.
    """
    names = [m.get("speaker_name") for m in mentions if m.get("speaker_name")]
    if not names:
        return None

    # Count occurrences
    name_counts: dict[str, int] = {}
    for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1

    # Sort by count (desc), then length (desc)
    sorted_names = sorted(
        name_counts.items(),
        key=lambda x: (x[1], len(x[0])),
        reverse=True,
    )

    return sorted_names[0][0]


def _build_profile_from_cluster(
    cluster: list[str],
    mentions_by_id: dict[str, list[dict]],
) -> SpeakerProfile:
    """Build a speaker profile from a cluster of IDs.

    Args:
        cluster: List of speaker IDs in this cluster.
        mentions_by_id: All mentions grouped by ID.

    Returns:
        SpeakerProfile for this cluster.
    """
    # Collect all mentions in cluster
    all_mentions = []
    for sid in cluster:
        all_mentions.extend(mentions_by_id.get(sid, []))

    # Get canonical name
    names = [m.get("speaker_name") for m in all_mentions if m.get("speaker_name")]
    canonical_name = _get_representative_name(all_mentions) or "Unknown Speaker"

    # Determine role by voting
    roles = [m.get("role", "unknown") for m in all_mentions]
    role_counts: dict[str, int] = {}
    for r in roles:
        role_counts[r] = role_counts.get(r, 0) + 1

    dominant_role_str = max(role_counts.items(), key=lambda x: x[1])[0]
    try:
        role = SpeakerRole(dominant_role_str)
    except ValueError:
        role = SpeakerRole.UNKNOWN

    # Generate canonical ID
    canonical_id = _generate_speaker_id(canonical_name)

    return SpeakerProfile(
        speaker_id=canonical_id,
        canonical_name=canonical_name,
        role=role,
        title=None,
        organization=None,
        mention_count=len(all_mentions),
    )


def _generate_speaker_id(name: str) -> str:
    """Generate a normalized speaker ID from name.

    Args:
        name: Speaker name.

    Returns:
        Normalized ID string.
    """
    import uuid

    if not name:
        return f"unknown_{uuid.uuid4().hex[:6]}"

    normalized = name.lower()
    normalized = "".join(c if c.isalnum() or c.isspace() else "" for c in normalized)
    normalized = "_".join(normalized.split())

    return normalized or f"speaker_{uuid.uuid4().hex[:6]}"
