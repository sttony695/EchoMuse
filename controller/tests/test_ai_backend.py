"""
The AI voice backend end to end: em_wyoming against a fake Wyoming server,
em_llm against a fake streaming chat API, and em_ai's turn against both with a
fake Echo feeding its voice queue.

Everything runs on 127.0.0.1 with real sockets, so what is tested is the wire,
not a mock of it.
"""

import asyncio
import collections
import contextlib
import json
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from aiohttp import web

import em_ai
import em_llm
import em_wyoming

FRAME = 2560                                    # 80 ms of 16 kHz S16 mono


def _loud(n=1):
    t = np.arange(1280 * n) / 16000
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


QUIET = bytes(FRAME)


# ── a fake Wyoming server ───────────────────────────────────────────────────

class FakeWyoming:
    """Speech to text answers `transcript`; text to speech answers 0.25 s of
    a tone per synthesize, at 22.05 kHz like Piper. Records what it got."""

    def __init__(self, transcript="what time is it", stt_delay=0.0,
                 inline_data=False, fail_tts=False):
        self.transcript = transcript
        self.stt_delay = stt_delay
        self.inline_data = inline_data
        self.fail_tts = fail_tts
        self.heard = bytearray()
        self.said: list[str] = []
        self.language = None

    def _event(self, etype, data=None, payload=b""):
        if not self.inline_data:
            return em_wyoming.encode(etype, data, payload)
        header = {"type": etype, "data": data or {}}
        if payload:
            header["payload_length"] = len(payload)
        return json.dumps(header).encode() + b"\n" + payload

    async def handle(self, reader, writer):
        try:
            while True:
                event = await em_wyoming.read_event(reader)
                if event.type == "transcribe":
                    self.language = event.data.get("language")
                elif event.type == "audio-chunk":
                    self.heard.extend(event.payload)
                elif event.type == "audio-stop":
                    await asyncio.sleep(self.stt_delay)
                    writer.write(self._event("transcript", {"text": self.transcript}))
                    await writer.drain()
                elif event.type == "synthesize":
                    self.said.append(event.data["text"])
                    if self.fail_tts:
                        writer.write(self._event("error", {"text": "no voice"}))
                        await writer.drain()
                        continue
                    fmt = {"rate": 22050, "width": 2, "channels": 1}
                    pcm = (np.ones(22050 // 4) * 1000).astype("<i2").tobytes()
                    writer.write(self._event("audio-start", fmt)
                                 + self._event("audio-chunk", fmt, pcm[:4000])
                                 + self._event("audio-chunk", fmt, pcm[4000:])
                                 + self._event("audio-stop", {}))
                    await writer.drain()
        except (em_wyoming.WyomingError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


@contextlib.asynccontextmanager
async def wyoming_server(fake):
    server = await asyncio.start_server(fake.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"tcp://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


# ── a fake OpenAI-compatible chat API ───────────────────────────────────────

class FakeLLM:
    def __init__(self, deltas=("It is ", "nine o'clock. ", "Anything ", "else?"),
                 status=200, delay=0.0):
        self.deltas = deltas
        self.status = status
        self.delay = delay
        self.requests: list[dict] = []
        self.auth = []

    async def chat(self, request):
        self.requests.append(await request.json())
        self.auth.append(request.headers.get("Authorization"))
        if self.status != 200:
            return web.Response(status=self.status, text="model not found")
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b": keep-alive\n\n")
        for d in self.deltas:
            await asyncio.sleep(self.delay)
            chunk = {"choices": [{"delta": {"content": d}}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp


@contextlib.asynccontextmanager
async def llm_server(fake):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake.chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


# ── em_wyoming ──────────────────────────────────────────────────────────────

def test_parse_uri():
    assert em_wyoming.parse_uri("tcp://192.168.1.115:10300") == ("192.168.1.115", 10300)
    for bad in ("http://x:1", "tcp://x", "x:1", ""):
        with pytest.raises(ValueError):
            em_wyoming.parse_uri(bad)


@pytest.mark.parametrize("inline", [False, True])
def test_transcribe_and_synthesize(inline):
    fake = FakeWyoming(inline_data=inline)

    async def go():
        async with wyoming_server(fake) as uri:
            text = await em_wyoming.transcribe(uri, _loud(3), language="en")
            audio = await em_wyoming.synthesize(uri, "Hello there.", voice="lessac")
        return text, audio
    text, audio = asyncio.run(go())
    assert text == "what time is it"
    assert bytes(fake.heard) == _loud(3) and fake.language == "en"
    assert (audio.rate, audio.width, audio.channels) == (22050, 2, 1)
    assert len(audio.pcm) == 2 * (22050 // 4)
    assert fake.said == ["Hello there."]


def test_a_server_error_is_raised():
    fake = FakeWyoming(fail_tts=True)

    async def go():
        async with wyoming_server(fake) as uri:
            await em_wyoming.synthesize(uri, "x")
    with pytest.raises(em_wyoming.WyomingError, match="no voice"):
        asyncio.run(go())


def test_an_unreachable_server_is_a_wyoming_error():
    async def go():
        server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        server.close()
        await server.wait_closed()
        await em_wyoming.transcribe(f"tcp://127.0.0.1:{port}", QUIET, timeout=2)
    with pytest.raises(em_wyoming.WyomingError, match="cannot reach"):
        asyncio.run(go())


# ── em_llm ──────────────────────────────────────────────────────────────────

def test_stream_reply_yields_deltas_and_sends_the_key():
    fake = FakeLLM()

    async def go():
        async with llm_server(fake) as url:
            return [d async for d in em_llm.stream_reply(
                url, "m1", [{"role": "user", "content": "hi"}], api_key="k")]
    assert asyncio.run(go()) == ["It is ", "nine o'clock. ", "Anything ", "else?"]
    assert fake.requests[0]["model"] == "m1" and fake.requests[0]["stream"] is True
    assert fake.auth == ["Bearer k"]


def test_a_refusal_is_an_llm_error():
    fake = FakeLLM(status=404)

    async def go():
        async with llm_server(fake) as url:
            return [d async for d in em_llm.stream_reply(url, "m", [])]
    with pytest.raises(em_llm.LLMError, match="404"):
        asyncio.run(go())


def test_chat_url():
    assert em_llm.chat_url("http://h/v1/") == "http://h/v1/chat/completions"
    assert em_llm.chat_url("http://h/v1/chat/completions") == "http://h/v1/chat/completions"


def test_sentences_regroup_deltas():
    async def deltas():
        for d in ["Pi is 3.", "14 or so. Is ", "that ok?", " Bye"]:
            yield d

    async def go():
        return [s async for s in em_llm.sentences(deltas())]
    assert asyncio.run(go()) == ["Pi is 3.14 or so.", "Is that ok?", "Bye"]


def test_system_prompt_carries_the_time():
    now = datetime(2026, 11, 22, 13, 55, tzinfo=timezone(timedelta(hours=11)))
    p = em_llm.system_prompt("", now)
    assert p.startswith(em_llm.SYSTEM_PROMPT)
    assert "Sunday 22 November 2026, 1:55 PM (UTC+11:00)" in p
    assert em_llm.system_prompt("Be terse.", now).startswith("Be terse. ")


# ── em_ai: one whole turn against a fake Echo ───────────────────────────────

class FakeDevice:
    def __init__(self, frames, device_id="G090LF0000000009"):
        self.device_id = device_id
        self.voice_queue: asyncio.Queue = asyncio.Queue()
        for f in frames:
            self.voice_queue.put_nowait(f)
        self.cancel_event = asyncio.Event()
        self.noise_floor = 0.0
        self.ns_asr = False
        self.save_utterances = True
        self.last_utterance_pcm = None
        self.last_wake = {"score": 0.8, "model": "hey_jarvis"}
        self.turn_history = collections.deque(maxlen=50)
        self.beam_locks = 0

    async def beam_lock(self):
        self.beam_locks += 1


@pytest.fixture
def turn_env(monkeypatch):
    """No Silero model (the plain-RMS path), a fast endpoint, and the turn
    record captured instead of written."""
    monkeypatch.setattr(em_ai.em_speechgate, "new_turn", lambda: None)
    monkeypatch.setattr(em_ai, "ENDPOINT_GRACE_S", 0.05)
    monkeypatch.setattr(em_ai, "ENDPOINT_SILENCE_S", 0.15)
    persisted = []

    async def persist(device, record):
        persisted.append(record)

    async def push(_event):
        return None
    monkeypatch.setattr(em_ai.em_turn_record, "persist_turn", persist)
    monkeypatch.setattr(em_ai.em_events, "push", push)
    monkeypatch.setattr(em_ai, "_devices", {})
    monkeypatch.setattr(em_ai, "_turns", {})
    return persisted


def run_turn(monkeypatch, device, wyoming, llm, *, during=None, model="m"):
    played = bytearray()
    thinking = []

    async def on_thinking():
        thinking.append(True)

    async def post_turn_play(chunks):
        n = 0
        async for chunk in chunks:
            if device.cancel_event.is_set():
                break
            played.extend(chunk)
            n += len(chunk)
        return n

    async def go():
        async with wyoming_server(wyoming) as wy, llm_server(llm) as url:
            monkeypatch.setattr(em_ai, "STT_URI", wy)
            monkeypatch.setattr(em_ai, "TTS_URI", wy)
            monkeypatch.setattr(em_ai, "LLM_URL", url)
            monkeypatch.setattr(em_ai, "LLM_MODEL", model)
            task = asyncio.create_task(em_ai.trigger_voice_turn(
                device, on_thinking, post_turn_play, trigger_label="wakeword(0.8)",
                preroll_discard=1))
            if during is not None:
                await during(device)
            went_on = await task
            await asyncio.sleep(0)               # let spawned on_thinking run
            return went_on
    went_on = asyncio.run(go())
    return went_on, bytes(played), thinking


def test_a_whole_turn(monkeypatch, turn_env):
    wy, llm = FakeWyoming(), FakeLLM()
    device = FakeDevice([QUIET] + [_loud()] * 5 + [QUIET] * 4)
    went_on, played, thinking = run_turn(monkeypatch, device, wy, llm)

    assert went_on is False and thinking == [True]
    # the first (preroll) frame was dropped; the rest reached speech to text
    assert len(wy.heard) == 9 * FRAME
    assert wy.said == ["It is nine o'clock.", "Anything else?"]
    # two 0.25 s sentences at the wire rate (5512 samples at 22.05 kHz each,
    # 11999 once resampled)
    assert abs(len(played) - 2 * 2 * 48000 // 4) <= 8
    record = turn_env[0]
    assert record["outcome"] == "ok" and record["stt_text"] == "what time is it"
    assert record["tts_bytes"] == len(played) and record["playback_ms"] >= 0
    # the model was told the time and got the question
    messages = llm.requests[0]["messages"]
    assert messages[0]["role"] == "system" and "current local date" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "what time is it"}
    # the conversation is remembered for a follow-up
    assert em_ai._devices[device.device_id].history[-1] == {
        "role": "assistant", "content": "It is nine o'clock. Anything else?"}
    assert device.last_utterance_pcm and device.beam_locks == 1
    assert device.device_id not in em_ai._turns


def test_a_follow_up_carries_the_last_exchange(monkeypatch, turn_env):
    wy, llm = FakeWyoming(), FakeLLM()
    run_turn(monkeypatch, FakeDevice([_loud()] * 4 + [QUIET] * 4), wy, llm)
    run_turn(monkeypatch, FakeDevice([_loud()] * 4 + [QUIET] * 4), wy, llm)
    second = llm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]


def test_the_echo_vad_sentinel_ends_the_command(monkeypatch, turn_env):
    monkeypatch.setattr(em_ai, "ENDPOINT_SILENCE_S", 30)
    device = FakeDevice([_loud()] * 3 + [em_ai.VAD_SENTINEL_END])
    run_turn(monkeypatch, device, FakeWyoming(), FakeLLM())
    assert turn_env[0]["outcome"] == "ok"


def test_no_speech(monkeypatch, turn_env):
    wy = FakeWyoming()
    device = FakeDevice([QUIET, QUIET, em_ai.VAD_SENTINEL_TIMEOUT])
    went_on, played, thinking = run_turn(monkeypatch, device, wy, FakeLLM())
    assert turn_env[0]["outcome"] == "no_speech"
    assert not wy.heard and not played and not thinking


def test_nothing_recognised(monkeypatch, turn_env):
    device = FakeDevice([_loud()] * 4 + [QUIET] * 4)
    run_turn(monkeypatch, device, FakeWyoming(transcript=""), FakeLLM())
    assert turn_env[0]["outcome"] == "no_speech"


def test_a_cancel_while_transcribing_ends_the_turn_at_once(monkeypatch, turn_env):
    wy = FakeWyoming(stt_delay=5.0)
    device = FakeDevice([_loud()] * 4 + [QUIET] * 4)

    async def barge(dev):
        await asyncio.sleep(0.6)
        dev.cancel_event.set()
        em_ai.cancel_voice_turn(dev.device_id, abort_ha=True, reason="barged")

    went_on, played, _ = run_turn(monkeypatch, device, wy, FakeLLM(), during=barge)
    assert turn_env[0]["outcome"] == "barged" and not played
    assert turn_env[0]["total_ms"] < 3000


def test_a_barge_during_playback_is_recorded(monkeypatch, turn_env):
    llm = FakeLLM(deltas=["One. ", "Two. ", "Three. ", "Four."], delay=0.2)
    device = FakeDevice([_loud()] * 4 + [QUIET] * 4)

    async def barge(dev):
        while not em_ai._turns.get(dev.device_id) or not llm.requests:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.5)
        em_ai.note_barged(dev.device_id)
        dev.cancel_event.set()

    went_on, played, _ = run_turn(monkeypatch, device, FakeWyoming(), llm, during=barge)
    assert turn_env[0]["outcome"] == "barged"
    assert 0 < len(played) < 4 * 2 * 48000 // 4


def test_a_model_failure_is_apologised_for(monkeypatch, turn_env):
    wy = FakeWyoming()
    device = FakeDevice([_loud()] * 4 + [QUIET] * 4)
    went_on, played, _ = run_turn(monkeypatch, device, wy, FakeLLM(status=500))
    assert turn_env[0]["outcome"] == "llm_error"
    assert wy.said == [em_ai.APOLOGY_NO_MODEL] and played


def test_a_voice_failure_is_a_tts_error(monkeypatch, turn_env):
    device = FakeDevice([_loud()] * 4 + [QUIET] * 4)
    went_on, played, _ = run_turn(monkeypatch, device, FakeWyoming(fail_tts=True), FakeLLM())
    assert turn_env[0]["outcome"] == "tts_error" and not played


def test_to_wire_resamples_and_mixes_down():
    stereo = np.zeros((22050, 2), dtype="<i2")
    stereo[:, 0] = 1000
    stereo[:, 1] = 3000
    pcm = em_ai.to_wire(em_wyoming.Audio(stereo.tobytes(), 22050, 2, 2))
    out = np.frombuffer(pcm, dtype="<i2")
    assert len(out) == 48000
    assert abs(int(out[24000]) - 2000) < 5
    same = em_ai.to_wire(em_wyoming.Audio(b"\x01\x00" * 10, 48000, 2, 1))
    assert same == b"\x01\x00" * 10
    with pytest.raises(em_wyoming.WyomingError):
        em_ai.to_wire(em_wyoming.Audio(b"\x00" * 9, 16000, 3, 1))
