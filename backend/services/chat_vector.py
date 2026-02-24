"""Vector search fallback for the chatbot using ChromaDB + Ollama embeddings.

Layer 2 retrieval: used only when structured search returns insufficient results.
All vector results map back to qa_id + page refs for citation grounding.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup cost if vector search isn't used
_chromadb = None
_ollama_ef = None


def _get_chromadb():
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


def _get_embedding_function():
    """Get Ollama embedding function using nomic-embed-text."""
    global _ollama_ef
    if _ollama_ef is None:
        import chromadb.utils.embedding_functions as ef
        _ollama_ef = ef.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
    return _ollama_ef


# Persistent storage directory for vector indices
VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"


def get_or_create_collection(run_id: str):
    """Get or create a ChromaDB collection for a specific run.

    Each run has its own collection to keep data isolated.
    """
    chromadb = _get_chromadb()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection_name = f"run_{run_id.replace('-', '_')[:50]}"

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
        metadata={"run_id": run_id},
    )


def index_run_data(run_id: str, run_data: dict) -> int:
    """Index Q&A units from a run into the vector store.

    Each Q&A unit is embedded as a combined question + answer block.
    Metadata includes qa_id, questioner_name, page numbers for grounding.

    Returns number of documents indexed.
    """
    collection = get_or_create_collection(run_id)

    # Check if already indexed
    existing = collection.count()
    if existing > 0:
        logger.info(f"Run {run_id} already indexed ({existing} documents)")
        return existing

    qa_units = run_data.get("qa", {}).get("qa_units", [])
    if not qa_units:
        logger.warning(f"No Q&A units to index for run {run_id}")
        return 0

    documents = []
    metadatas = []
    ids = []

    for qa in qa_units:
        qa_id = qa["qa_id"]
        questioner = qa.get("questioner_name", "Unknown")
        question_text = qa.get("question_text", "")
        response_text = qa.get("response_text", "")
        start_page = qa.get("start_page", 0)
        end_page = qa.get("end_page", 0)
        responders = ", ".join(qa.get("responder_names", []))

        # Combined document for embedding
        doc = f"Question by {questioner}: {question_text}\nAnswer by {responders}: {response_text}"

        documents.append(doc)
        metadatas.append({
            "qa_id": qa_id,
            "questioner_name": questioner,
            "responder_names": responders,
            "start_page": start_page,
            "end_page": end_page,
            "is_follow_up": str(qa.get("is_follow_up", False)),
            "source_type": "structured_qa",
        })
        ids.append(f"{run_id}_{qa_id}")

    # Index in batches (ChromaDB has a batch limit)
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )

    logger.info(f"Indexed {len(documents)} Q&A units for run {run_id}")
    return len(documents)


def vector_search(run_id: str, query: str, n_results: int = 5) -> str:
    """Search the vector store for semantically similar Q&A units.

    Returns results mapped back to qa_ids and page refs.
    """
    try:
        collection = get_or_create_collection(run_id)

        if collection.count() == 0:
            return json.dumps({
                "results": [],
                "error": "Vector index is empty. Run indexing first.",
                "source": "vector_search",
            })

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

        formatted = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None

                formatted.append({
                    "qa_id": meta.get("qa_id", "unknown"),
                    "questioner_name": meta.get("questioner_name", "unknown"),
                    "responder_names": meta.get("responder_names", ""),
                    "start_page": meta.get("start_page", 0),
                    "end_page": meta.get("end_page", 0),
                    "text": doc[:1000],
                    "similarity_distance": round(distance, 4) if distance else None,
                    "source": "vector_search",
                })

        return json.dumps({
            "results": formatted,
            "total": len(formatted),
            "query": query,
            "source": "vector_search",
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return json.dumps({
            "results": [],
            "error": str(e),
            "source": "vector_search",
        })


def delete_run_index(run_id: str) -> bool:
    """Delete the vector index for a specific run."""
    try:
        chromadb = _get_chromadb()
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        collection_name = f"run_{run_id.replace('-', '_')[:50]}"
        client.delete_collection(collection_name)
        logger.info(f"Deleted vector index for run {run_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete vector index for run {run_id}: {e}")
        return False
