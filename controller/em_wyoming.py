"""
em_wyoming.py — a minimal Wyoming client, for em_ai's speech to text and text
to speech.

Wyoming (github.com/OHF-Voice/wyoming) is the protocol Home Assistant's own
Whisper, Piper and Kokoro add-ons speak, so the AI backend can use the same
servers a user already runs. An event on the wire is one JSON header line,
then `data_length` bytes of JSON data, then `payload_length` bytes of binary
payload. Older peers put `data` inline in the header instead; both are read.

Only the two exchanges em_ai needs:

  transcribe   transcribe {language} → audio-start → audio-chunk… → audio-stop
               ← transcript {text}
  synthesize   synthesize {text, voice}
               ← audio-start {rate, width, channels} → audio-chunk… → audio-stop

Pure asyncio, no third-party package, so it can be tested against a fake
server in the suite.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlparse

PROTOCOL_VERSION = "1.5.3"
CHUNK_BYTES = 2048          # audio sent to the server per audio-chunk
MAX_LINE = 1 << 20          # a header line longer than this is not Wyoming


class WyomingError(Exception):
    """The server could not be reached, refused, or broke the protocol."""


@dataclass
class Event:
    type: str
    data: dict
    payload: bytes = b""


def parse_uri(uri: str) -> tuple[str, int]:
    """`tcp://host:port` → (host, port). Wyoming servers listen on TCP."""
    parsed = urlparse(uri.strip())
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
        raise ValueError(f"not a Wyoming tcp://host:port URI: {uri!r}")
    return parsed.hostname, parsed.port


def encode(event_type: str, data: dict | None = None, payload: bytes = b"") -> bytes:
    data_bytes = json.dumps(data or {}, ensure_ascii=False).encode("utf-8")
    header = {"type": event_type, "version": PROTOCOL_VERSION,
              "data_length": len(data_bytes)}
    if payload:
        header["payload_length"] = len(payload)
    return json.dumps(header).encode("utf-8") + b"\n" + data_bytes + payload


async def read_event(reader: asyncio.StreamReader) -> Event:
    line = await reader.readline()
    if not line:
        raise WyomingError("connection closed")
    if len(line) > MAX_LINE or not line.endswith(b"\n"):
        raise WyomingError("malformed header line")
    try:
        header = json.loads(line)
    except ValueError as e:
        raise WyomingError(f"header is not JSON: {e}") from None
    data = dict(header.get("data") or {})
    if header.get("data_length"):
        extra = await reader.readexactly(int(header["data_length"]))
        data.update(json.loads(extra))
    payload = b""
    if header.get("payload_length"):
        payload = await reader.readexactly(int(header["payload_length"]))
    return Event(str(header.get("type", "")), data, payload)


async def _open(uri: str, timeout: float):
    host, port = parse_uri(uri)
    try:
        return await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError) as e:
        raise WyomingError(f"cannot reach {uri}: {e or type(e).__name__}") from None


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


async def transcribe(uri: str, pcm: bytes, language: str = "",
                     rate: int = 16000, timeout: float = 30.0) -> str:
    """Speech to text for 16-bit mono PCM. Returns the transcript ("" when
    the server heard nothing)."""
    reader, writer = await _open(uri, timeout)
    try:
        audio = {"rate": rate, "width": 2, "channels": 1}
        out = [encode("transcribe", {"language": language} if language else {}),
               encode("audio-start", audio)]
        for i in range(0, len(pcm), CHUNK_BYTES):
            out.append(encode("audio-chunk", audio, pcm[i:i + CHUNK_BYTES]))
        out.append(encode("audio-stop", {}))
        writer.write(b"".join(out))
        await writer.drain()

        async def _wait() -> str:
            while True:
                event = await read_event(reader)
                if event.type == "transcript":
                    return str(event.data.get("text", "")).strip()
                if event.type == "error":
                    raise WyomingError(f"server error: {event.data.get('text', '')}")
        return await asyncio.wait_for(_wait(), timeout)
    except asyncio.TimeoutError:
        raise WyomingError(f"no transcript from {uri} within {timeout:.0f}s") from None
    except (asyncio.IncompleteReadError, ConnectionError) as e:
        raise WyomingError(f"connection to {uri} failed: {e}") from None
    finally:
        await _close(writer)


@dataclass
class Audio:
    pcm: bytes
    rate: int
    width: int
    channels: int


async def synthesize(uri: str, text: str, voice: str = "",
                     timeout: float = 30.0) -> Audio:
    """Text to speech. Returns the whole utterance; em_ai calls this one
    sentence at a time, so the first sentence plays while the rest are made."""
    reader, writer = await _open(uri, timeout)
    try:
        data: dict = {"text": text}
        if voice:
            data["voice"] = {"name": voice}
        writer.write(encode("synthesize", data))
        await writer.drain()

        async def _collect() -> Audio:
            fmt = None
            chunks: list[bytes] = []
            while True:
                event = await read_event(reader)
                if event.type == "audio-start":
                    fmt = event.data
                elif event.type == "audio-chunk":
                    fmt = fmt or event.data
                    chunks.append(event.payload)
                elif event.type == "audio-stop":
                    break
                elif event.type == "error":
                    raise WyomingError(f"server error: {event.data.get('text', '')}")
            if fmt is None:
                raise WyomingError("audio-stop with no audio format")
            return Audio(b"".join(chunks), int(fmt.get("rate", 22050)),
                         int(fmt.get("width", 2)), int(fmt.get("channels", 1)))
        return await asyncio.wait_for(_collect(), timeout)
    except asyncio.TimeoutError:
        raise WyomingError(f"no audio from {uri} within {timeout:.0f}s") from None
    except (asyncio.IncompleteReadError, ConnectionError) as e:
        raise WyomingError(f"connection to {uri} failed: {e}") from None
    finally:
        await _close(writer)
