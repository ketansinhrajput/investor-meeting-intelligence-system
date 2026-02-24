"""Topic aggregation utilities."""

from collections import defaultdict

from src.models import TopicSummary


def aggregate_topics(
    enriched_qa_units: list[dict],
    enriched_strategic: list[dict],
) -> list[TopicSummary]:
    """Aggregate topics across all units and statements.

    Args:
        enriched_qa_units: List of enriched Q&A unit dicts.
        enriched_strategic: List of enriched strategic statement dicts.

    Returns:
        List of TopicSummary objects sorted by mention count.
    """
    topic_data: dict[str, dict] = defaultdict(lambda: {
        "topic_name": "",
        "topic_category": "General",
        "qa_unit_ids": [],
        "statement_ids": [],
        "evidence_spans": [],
        "sentiments": [],
    })

    # Collect topics from Q&A units
    for qa in enriched_qa_units:
        unit_id = qa.get("unit_id", "")
        intent = qa.get("investor_intent", {})
        posture = qa.get("response_posture", {})

        for topic in qa.get("topics", []):
            topic_name = topic.get("topic_name", "").lower().strip()
            if not topic_name:
                continue

            topic_data[topic_name]["topic_name"] = topic.get("topic_name", topic_name)
            topic_data[topic_name]["topic_category"] = topic.get("topic_category", "General")
            topic_data[topic_name]["qa_unit_ids"].append(unit_id)
            topic_data[topic_name]["evidence_spans"].extend(
                topic.get("evidence_spans", [])
            )

            # Track sentiment based on posture
            posture_value = posture.get("primary_posture", "neutral")
            topic_data[topic_name]["sentiments"].append(
                _posture_to_sentiment(posture_value)
            )

    # Collect topics from strategic statements
    for stmt in enriched_strategic:
        statement_id = stmt.get("statement_id", "")
        sentiment = stmt.get("sentiment", "neutral")

        for topic in stmt.get("topics", []):
            topic_name = topic.get("topic_name", "").lower().strip()
            if not topic_name:
                continue

            topic_data[topic_name]["topic_name"] = topic.get("topic_name", topic_name)
            topic_data[topic_name]["topic_category"] = topic.get("topic_category", "General")
            topic_data[topic_name]["statement_ids"].append(statement_id)
            topic_data[topic_name]["evidence_spans"].extend(
                topic.get("evidence_spans", [])
            )
            topic_data[topic_name]["sentiments"].append(sentiment)

    # Build topic summaries
    summaries = []
    for topic_name, data in topic_data.items():
        qa_ids = list(set(data["qa_unit_ids"]))
        stmt_ids = list(set(data["statement_ids"]))
        mention_count = len(qa_ids) + len(stmt_ids)

        # Calculate sentiment distribution
        sentiment_dist = _calculate_sentiment_distribution(data["sentiments"])

        # Generate summary text
        summary_text = _generate_topic_summary(
            topic_name=data["topic_name"],
            qa_count=len(qa_ids),
            stmt_count=len(stmt_ids),
            sentiment_dist=sentiment_dist,
        )

        # Get unique key points (evidence spans)
        key_points = list(set(data["evidence_spans"]))[:5]

        summary = TopicSummary(
            topic_name=data["topic_name"],
            topic_category=data["topic_category"],
            mention_count=mention_count,
            qa_unit_ids=qa_ids,
            statement_ids=stmt_ids,
            summary=summary_text,
            key_points=key_points,
            sentiment_distribution=sentiment_dist,
        )
        summaries.append(summary)

    # Sort by mention count (descending)
    summaries.sort(key=lambda x: x.mention_count, reverse=True)

    return summaries


def _posture_to_sentiment(posture: str) -> str:
    """Map response posture to sentiment.

    Args:
        posture: Response posture value.

    Returns:
        Sentiment string.
    """
    positive_postures = {"confident", "optimistic", "transparent"}
    negative_postures = {"defensive", "cautious", "evasive"}

    if posture in positive_postures:
        return "positive"
    elif posture in negative_postures:
        return "negative"
    else:
        return "neutral"


def _calculate_sentiment_distribution(sentiments: list[str]) -> dict[str, int]:
    """Calculate sentiment distribution from list.

    Args:
        sentiments: List of sentiment strings.

    Returns:
        Dict mapping sentiment to count.
    """
    dist: dict[str, int] = {}
    for s in sentiments:
        dist[s] = dist.get(s, 0) + 1
    return dist


def _generate_topic_summary(
    topic_name: str,
    qa_count: int,
    stmt_count: int,
    sentiment_dist: dict[str, int],
) -> str:
    """Generate a summary sentence for a topic.

    Args:
        topic_name: Name of the topic.
        qa_count: Number of Q&A units discussing this topic.
        stmt_count: Number of strategic statements with this topic.
        sentiment_dist: Sentiment distribution.

    Returns:
        Summary sentence.
    """
    parts = []

    # Mention counts
    if qa_count > 0 and stmt_count > 0:
        parts.append(
            f"'{topic_name}' was discussed in {qa_count} Q&A exchange(s) "
            f"and {stmt_count} strategic statement(s)."
        )
    elif qa_count > 0:
        parts.append(f"'{topic_name}' was discussed in {qa_count} Q&A exchange(s).")
    else:
        parts.append(f"'{topic_name}' appeared in {stmt_count} strategic statement(s).")

    # Dominant sentiment
    if sentiment_dist:
        dominant = max(sentiment_dist.items(), key=lambda x: x[1])
        if dominant[0] != "neutral":
            parts.append(f"Overall sentiment appears {dominant[0]}.")

    return " ".join(parts)
