"""Chat endpoint for conversational Q&A over analysis runs.

POST /runs/{run_id}/chat - Ask a question (returns full response).
POST /runs/{run_id}/chat/stream - Ask a question (streams SSE tokens).
Uses 2-phase approach: structured retrieval + grounded LLM synthesis.
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.requests import ChatRequest
from api.schemas.responses import ChatCitation, ChatResponse, ChatToolCall
from services.chat_agent import chat as chat_agent, chat_stream as chat_agent_stream
from services.chat_data_loader import load_run_data

logger = logging.getLogger(__name__)

router = APIRouter()

RUNS_DIR = Path(__file__).parent.parent.parent / "data" / "runs"


def _validate_run(run_id: str) -> dict:
    """Validate run exists and is completed, return run_data."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    try:
        run_data = load_run_data(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run data not found: {run_id}")

    status = run_data.get("metadata", {}).get("status", "unknown")
    if status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed (status: {status}). Chat is only available for completed runs.",
        )

    return run_data


@router.post("/runs/{run_id}/chat", response_model=ChatResponse)
async def chat_with_run(run_id: str, request: ChatRequest):
    """Chat with an analyzed transcript run (full response)."""
    run_data = _validate_run(run_id)

    try:
        result = chat_agent(
            question=request.message,
            run_data=run_data,
            history=[{"role": m.role, "content": m.content} for m in request.history],
        )
    except Exception as e:
        logger.error(f"Chat agent error for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

    return ChatResponse(
        run_id=run_id,
        answer=result.answer,
        citations=[
            ChatCitation(type=c["type"], ref_id=c["ref_id"], label=c["label"])
            for c in result.citations
        ],
        tool_calls=[
            ChatToolCall(tool=tc["tool"], params=tc.get("params", {}))
            for tc in result.tool_calls
        ],
        retrieval_source=result.retrieval_source,
        total_time_seconds=result.total_time_seconds,
        model=result.model,
        disclaimer=result.disclaimer,
    )


@router.post("/runs/{run_id}/chat/stream")
async def chat_stream_with_run(run_id: str, request: ChatRequest):
    """Chat with an analyzed transcript run (streaming SSE).

    Returns Server-Sent Events:
      event: metadata  — {tool_calls, retrieval_source}
      event: token     — {text: "chunk"}
      event: done      — {citations, total_time_seconds, disclaimer, model}
    """
    run_data = _validate_run(run_id)

    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def sse_generator():
        """Wrap synchronous chat_stream generator in async.

        Uses a queue to bridge sync generator (in thread) with async yields.
        """
        import json as _json
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _run_sync():
            try:
                gen = chat_agent_stream(
                    question=request.message,
                    run_data=run_data,
                    history=history,
                )
                for event_str in gen:
                    queue.put_nowait(event_str)
            except Exception as e:
                logger.error(f"Chat stream error for run {run_id}: {e}", exc_info=True)
                queue.put_nowait(f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n")
            finally:
                queue.put_nowait(sentinel)

        # Start sync generator in thread
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_sync)

        # Yield events as they arrive
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
