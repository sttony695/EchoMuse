"""
em_llm.py — a streaming client for any OpenAI-compatible chat API, for em_ai.

`POST {base}/chat/completions` with `stream: true` is the one call: it is
spoken by OpenAI, OpenRouter, Ollama, LM Studio, vLLM, LiteLLM and the
CLIProxyAPI-style gateways that put Claude or Gemini behind the same shape,
so one client reaches all of them. The reply arrives as server-sent events,
each `data: {json}` carrying a delta; `data: [DONE]` ends it.

`sentences()` regroups the deltas into whole sentences, so em_ai can start
speaking the first while the model is still writing the rest.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import AsyncIterator

import aiohttp

SYSTEM_PROMPT = (
    "You are a voice assistant answering through a small smart speaker. "
    "Reply in one to three short spoken sentences. Use plain conversational "
    "text only: no markdown, lists, emoji, code or links, because every word "
    "is read aloud."
)


class LLMError(Exception):
    """The endpoint could not be reached, refused, or sent something else."""


def system_prompt(base: str = "", now: datetime | None = None) -> str:
    """The instructions, with the local date and time: a model has no clock,
    and "what time is it" is one of the first things anyone asks a speaker."""
    now = now or datetime.now()
    if now.tzinfo is None:          # local time, given its offset
        now = now.astimezone()
    offset = now.strftime("%z")
    stamp = (f"{now:%A} {now.day} {now:%B %Y}, "
             f"{now.strftime('%I:%M %p').lstrip('0')} "
             f"(UTC{offset[:3]}:{offset[3:]})")
    return f"{base or SYSTEM_PROMPT} The current local date and time is {stamp}."


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


async def stream_reply(base_url: str, model: str, messages: list[dict],
                       api_key: str = "", max_tokens: int = 300,
                       timeout: float = 30.0,
                       session: aiohttp.ClientSession | None = None,
                       ) -> AsyncIterator[str]:
    """Yield the reply's text deltas. `timeout` bounds the wait for the
    response to begin; a reply already streaming is not cut off by it."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "stream": True,
            "max_tokens": max_tokens}
    own = session is None
    if own:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=timeout,
                                          sock_read=timeout))
    try:
        try:
            resp = await session.post(chat_url(base_url), json=body, headers=headers)
        except (aiohttp.ClientError, TimeoutError) as e:
            raise LLMError(f"cannot reach {base_url}: {e or type(e).__name__}") from None
        async with resp:
            if resp.status != 200:
                detail = (await resp.text())[:200]
                raise LLMError(f"{base_url} answered {resp.status}: {detail}")
            try:
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        raise LLMError(f"{base_url}: {chunk['error']}")
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text
            except (aiohttp.ClientError, TimeoutError) as e:
                raise LLMError(f"reply from {base_url} broke off: "
                               f"{e or type(e).__name__}") from None
    finally:
        if own:
            await session.close()


# A sentence ends at . ! ? or … (and any closing quote or bracket), followed by
# whitespace. A full stop between digits ("3.5") is not followed by a space.
_SENTENCE_END = re.compile(r"(.+?[.!?…]+[\"')\]]*)(\s+)", re.S)


async def sentences(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup streamed text into sentences, ending with whatever is left."""
    pending = ""
    async for delta in deltas:
        pending += delta
        while (match := _SENTENCE_END.match(pending)) is not None:
            sentence = match.group(1).strip()
            pending = pending[match.end():]
            if sentence:
                yield sentence
    if pending.strip():
        yield pending.strip()
