"""
em_ai.py — the controller answering its Echoes itself, with no Home Assistant.

Selected by VOICE_BACKEND=ai (em_voice.py). Exports the same names as
em_esphome, the Home Assistant backend, so em_controller and the route modules
drive either one through the same calls; tests/test_voice_backend.py holds the
two to that.

A turn, start to finish (trigger_voice_turn, voice_lock held by the caller):

  1. capture     the Echo's audio from device.voice_queue until the speaker
                 stops — the same pieces em_esphome's stream uses: the Silero
                 speech gate (em_speechgate), the DTLN denoiser when nsAsr is
                 on (em_ns), the no-speech rule (em_turnclock), utterance
                 capture for the Activity tab (saveUtterances) and the Echo's
                 own VAD sentinels. Home Assistant's VAD is the end-of-turn
                 authority on that backend; here there is none, so the
                 controller's own endpoint (ha_vad_stalled_verdict, with its
                 grace and silence set for a backend that has nothing else)
                 ends the command.
  2. thinking    on_thinking() — the ring's spinner, as at HA's STT_VAD_END.
  3. transcribe  Wyoming speech to text (AI_STT_URI; Whisper, as HA's own).
  4. reply       an OpenAI-compatible chat API (AI_LLM_URL / AI_LLM_MODEL),
                 streamed; each sentence goes to Wyoming text to speech
                 (AI_TTS_URI) as soon as it is complete, is converted to the
                 wire format (mono 48 kHz S16) and handed to post_turn_play,
                 so speech starts while the model is still writing.
  5. record      the turn's trace, persisted exactly as em_esphome does.

Cancelling (a barge, the button, mute) sets device.cancel_event and calls
cancel_voice_turn: capture and the transcription wait stop at once, and the
playback path stops pulling audio, which closes the reply's stream and with
it the request to the model.

Each Echo keeps its own short conversation, so a follow-up question has the
context of the last few exchanges.

The rest of em_esphome's surface:

  Device lifecycle   device_connected / device_disconnected / device_deleted,
                     set_device_capabilities — remembered per device; there
                     is no listener or mDNS record to bring up.
  Home Assistant     send_button_event, update_ambient_lux,
  entities           update_temperatures, push_media_state, update_oww_model,
                     update_device_volume — there is no Home Assistant to tell;
                     the button event is logged, the rest are no-ops.
  Picker and mute    get_mic_muted_and_wake_word, update_mic_muted,
                     update_wake_word — kept here, as em_esphome keeps them on
                     its server object.
  Output mute        output_mute_report / output_mute_on_reconnect — HA's
                     media-player mute does not exist without HA, so a volume
                     report is always kept and nothing is re-sent.
  Timers             clear_timers, dismiss_timer_alarm — none yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

import em_events
import em_llm
import em_ns
import em_recordings
import em_speechgate
import em_tasks
import em_turn_record
import em_turnclock
import em_wyoming
# Shared with the Home Assistant backend so both mean the same thing: the
# sentinels em_controller puts on a device's voice queue, the preroll a
# controller-scored wake discards, the hold that makes a press a long press,
# and the trigger label of a conversation started without a wake word.
from em_esphome import (  # noqa: F401 — re-exported through em_voice
    BUTTON_HOLD_MS,
    CONVERSATION_TRIGGER,
    VAD_SENTINEL_END,
    VAD_SENTINEL_TIMEOUT,
    VOICE_PREROLL_DISCARD,
)

log = logging.getLogger("echomuse.ai")

# ── Settings (read once at import, like the rest of the controller) ─────────
STT_URI = os.environ.get("AI_STT_URI", "").strip()
STT_LANGUAGE = os.environ.get("AI_STT_LANGUAGE", "en").strip()
TTS_URI = os.environ.get("AI_TTS_URI", "").strip()
TTS_VOICE = os.environ.get("AI_TTS_VOICE", "").strip()
LLM_URL = os.environ.get("AI_LLM_URL", "").strip()
LLM_MODEL = os.environ.get("AI_LLM_MODEL", "").strip()
LLM_KEY = os.environ.get("AI_LLM_KEY", "").strip()
LLM_PROMPT = os.environ.get("AI_SYSTEM_PROMPT", "").strip()


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


LLM_MAX_TOKENS = _positive_int(os.environ.get("AI_MAX_TOKENS"), 300)
# How long the reply may take to begin: the same setting, and the same
# ceiling, as Home Assistant's reply wait on the other backend.
REPLY_WAIT = em_turnclock.tts_wait_seconds(os.environ.get("EM_TTS_WAIT_TIMEOUT"))

STT_TIMEOUT = 30.0
TTS_TIMEOUT = 30.0
CAPTURE_CAP_S = 20.0        # the whole listening phase, as em_esphome's cap
POLL_S = 0.1                # how often the endpoint is re-checked
# The controller's own endpoint. On the HA backend it is a fallback that
# waits HA_VAD_GRACE for HA's VAD to engage; here it is the only one, so it
# may end a short command as soon as the speaker has paused.
ENDPOINT_GRACE_S = 0.8
ENDPOINT_SILENCE_S = 0.9
HISTORY_MESSAGES = 10       # user + assistant messages kept per Echo
WIRE_RATE = 48000
WIRE_CHUNK = 9600           # bytes handed to playback at a time (0.1 s)

APOLOGY_UNHEARD = "Sorry, I couldn't make that out."
APOLOGY_NO_MODEL = "Sorry, I can't reach the assistant right now."


def missing_settings() -> list[str]:
    """The settings a turn cannot run without, that are unset."""
    need = {"AI_STT_URI": STT_URI, "AI_TTS_URI": TTS_URI,
            "AI_LLM_URL": LLM_URL, "AI_LLM_MODEL": LLM_MODEL}
    return [name for name, value in need.items() if not value]


@dataclass
class _DeviceState:
    """What em_esphome keeps on a DeviceESPhomeServer, minus the server."""
    capabilities: list[str] = field(default_factory=list)
    mic_muted: bool = False
    wake_word_enabled: bool = True
    volume: Optional[float] = None
    connected: bool = False
    standalone_play: object = None
    send_volume_set: object = None
    ring_alarm: object = None
    stop_alarm: object = None
    start_conversation: object = None
    set_wake_word: object = None
    history: list[dict] = field(default_factory=list)


@dataclass
class _Turn:
    end_reason: Optional[str] = None     # first reason wins, as em_esphome
    stop: asyncio.Event = field(default_factory=asyncio.Event)


_devices: dict[str, _DeviceState] = {}
_turns: dict[str, _Turn] = {}


def _state(device_id: str) -> _DeviceState:
    return _devices.setdefault(device_id, _DeviceState())


# ─── Lifecycle ───────────────────────────────────────────────────────────────

async def start_esphome_servers(devices: dict, host: str = "0.0.0.0") -> None:
    """Nothing to listen on: the AI backend has no satellites for HA to dial."""
    missing = missing_settings()
    if missing:
        log.warning("Voice backend: AI (VOICE_BACKEND=ai) — not configured, "
                    f"no turn can be answered until {', '.join(missing)} "
                    f"{'is' if len(missing) == 1 else 'are'} set")
    else:
        log.info(f"Voice backend: AI (VOICE_BACKEND=ai) — speech {STT_URI}, "
                 f"voice {TTS_URI}, model {LLM_MODEL} at {LLM_URL}")


async def stop_esphome_servers() -> None:
    _devices.clear()
    _turns.clear()


async def device_connected(
    device_id: str,
    host: str = "0.0.0.0",
    standalone_play=None,
    send_volume_set=None,
    ring_alarm=None,
    stop_alarm=None,
    start_conversation=None,
    set_wake_word=None,
    wake_word_enabled: bool = True,
) -> None:
    s = _state(device_id)
    s.connected = True
    s.standalone_play = standalone_play
    s.send_volume_set = send_volume_set
    s.ring_alarm = ring_alarm
    s.stop_alarm = stop_alarm
    s.start_conversation = start_conversation
    s.set_wake_word = set_wake_word
    s.wake_word_enabled = bool(wake_word_enabled)


async def device_disconnected(device_id: str) -> None:
    s = _devices.get(device_id)
    if s is None:
        return
    s.connected = False
    s.send_volume_set = s.ring_alarm = s.stop_alarm = None
    s.start_conversation = s.set_wake_word = None


async def device_deleted(device_id: str) -> None:
    _devices.pop(device_id, None)
    _turns.pop(device_id, None)


def set_device_capabilities(device_id: str, caps: list[str]) -> None:
    _state(device_id).capabilities = list(caps or [])


def get_server(device_id: str) -> None:
    """There is no per-device server in this backend."""
    return None


def get_status(device_id: str) -> Optional[dict]:
    """The dashboard's satellite readout. None, as for a device with no
    satellite: there is no Home Assistant connection to report."""
    return None


# ─── Voice turns ─────────────────────────────────────────────────────────────

def can_serve_turn(device_id: str) -> bool:
    """Whether a turn started now could be answered: the settings it needs
    are all present. An Echo that cannot answer must stand down before it
    claims a wake, so one that could is not silenced by it."""
    return not missing_settings()


async def record_dropped_wake(device, trigger_label: str, wake_info) -> None:
    await em_turn_record.record_dropped_turn(device, trigger_label, wake_info)


async def trigger_voice_turn(
    device,
    on_thinking,
    post_turn_play,
    trigger_label: str = "unknown",
    preroll_discard: int = VOICE_PREROLL_DISCARD,
) -> bool:
    """Run one voice turn; return whether the conversation continues (never,
    yet: a follow-up needs the wake word or the button)."""
    wake_info = device.last_wake
    device.last_wake = None
    if missing_settings():
        log.warning(f"[{device.device_id}] AI backend not configured "
                    f"({', '.join(missing_settings())} unset) — turn dropped")
        await em_turn_record.record_dropped_turn(device, trigger_label, wake_info)
        return False

    trace = em_turn_record.TurnTrace(
        trigger=trigger_label, t0=time.monotonic(), wake_info=wake_info)
    turn = _Turn()
    _turns[device.device_id] = turn
    tag = f"ai.{device.device_id[-8:]}"
    try:
        audio = await capture(device, preroll_discard, trace, turn, tag)
        if audio is None:
            return False
        em_tasks.spawn(on_thinking())

        try:
            text = await _unless_stopped(device, turn, em_wyoming.transcribe(
                STT_URI, audio, STT_LANGUAGE, timeout=STT_TIMEOUT))
        except em_wyoming.WyomingError as e:
            log.error(f"[{tag}] speech to text failed: {e}")
            trace.outcome = "stt_error"
            await _play_line(device, trace, post_turn_play, APOLOGY_UNHEARD, tag)
            return False
        if text is None:
            trace.outcome = turn.end_reason or "cancelled"
            return False
        trace.t_stt_ms = trace.elapsed_ms()
        trace.stt_text = text
        if not text:
            log.info(f"[{tag}] nothing recognised")
            trace.outcome = "no_speech"
            return False
        log.info(f"[{tag}] heard: {text}")

        history = _state(device.device_id).history
        spoken: list[str] = []
        trace.t_playback_ms = trace.elapsed_ms()
        try:
            async with contextlib.aclosing(
                    _reply_audio(text, history, spoken, trace, tag)) as chunks:
                trace.tts_bytes = await post_turn_play(chunks) or 0
        except em_wyoming.WyomingError as e:
            log.error(f"[{tag}] text to speech failed: {e}")
            trace.outcome = "tts_error"
        finally:
            _remember(history, text, " ".join(spoken))

        if trace.outcome:
            pass                                    # set inside the reply
        elif turn.end_reason or device.cancel_event.is_set():
            trace.outcome = turn.end_reason or "cancelled"
        elif not spoken:
            trace.outcome = "silent_reply"
        else:
            trace.outcome = "ok"
        return False
    finally:
        if _turns.get(device.device_id) is turn:
            _turns.pop(device.device_id, None)
        await _persist(device, trace)
        log.info(f"[{tag}] turn complete ({trace.outcome})")


def cancel_voice_turn(device_id: str, abort_ha: bool = False,
                      reason: str = "cancelled") -> None:
    turn = _turns.get(device_id)
    if turn is None:
        return
    if turn.end_reason is None:
        turn.end_reason = reason
    turn.stop.set()


def abort_ha_run(device_id: str) -> None:
    """A barge during playback: the reply was delivered, so the turn is not
    cancelled; there is no pipeline run to serialise against."""
    note_barged(device_id)


def note_barged(device_id: str) -> None:
    turn = _turns.get(device_id)
    if turn is not None and turn.end_reason is None:
        turn.end_reason = "barged"


async def _unless_stopped(device, turn: _Turn, coro):
    """Await `coro`, or None as soon as the turn is cancelled."""
    task = asyncio.ensure_future(coro)
    stop = asyncio.ensure_future(turn.stop.wait())
    cancel = asyncio.ensure_future(device.cancel_event.wait())
    try:
        await asyncio.wait([task, stop, cancel], return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.cancel()
        cancel.cancel()
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    return None


# ─── 1. Capture ──────────────────────────────────────────────────────────────

async def capture(device, preroll_discard: int, trace, turn: _Turn,
                  tag: str = "ai") -> Optional[bytes]:
    """The command, from device.voice_queue. None when there is nothing to
    answer — cancelled, nobody spoke, or the stream never came — with
    trace.outcome saying which."""
    audio = bytearray()
    recording = bytearray() if getattr(device, "save_utterances", False) else None
    device.last_utterance_pcm = None

    denoiser = None
    if getattr(device, "ns_asr", False) and em_ns.available():
        try:
            denoiser = em_ns.StreamingDenoiser()
        except Exception as e:
            log.warning(f"[{tag}] NS init failed ({e}) — raw audio")

    vad = em_speechgate.new_turn()
    gate = em_speechgate.SpeechGate() if vad is not None else None
    loop = asyncio.get_running_loop()

    preroll_remaining = preroll_discard
    speech_seen = False
    listening_since = None
    first_speech_at = last_speech_at = None
    turn_start = time.monotonic()

    def _is_speech(chunk: bytes) -> bool:
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean((samples.astype(np.float64) / 32768.0) ** 2)))
        return rms >= max(3.0 * getattr(device, "noise_floor", 0.0), 0.004)

    def _end(outcome: str | None) -> Optional[bytes]:
        if trace.t_vad_end_ms == -1:
            trace.t_vad_end_ms = trace.elapsed_ms()
        if outcome is not None:
            trace.outcome = outcome
            return None
        return bytes(audio)

    try:
        while True:
            now = time.monotonic()
            if device.cancel_event.is_set() or turn.stop.is_set():
                return _end(turn.end_reason or "cancelled")
            if now - turn_start > CAPTURE_CAP_S:
                log.warning(f"[{tag}] listening hit the {CAPTURE_CAP_S:.0f}s cap")
                return _end(None if speech_seen else "stream_timeout")

            give_up, why = em_turnclock.no_speech_verdict(
                now=now, turn_start=turn_start,
                listening_since=listening_since, speech_seen=speech_seen)
            if give_up:
                log.info(f"[{tag}] {why}")
                return _end("no_speech")

            ended, _ = em_turnclock.ha_vad_stalled_verdict(
                now=now, speech_seen=speech_seen, ha_vad_started=False,
                first_speech_at=first_speech_at, last_speech_at=last_speech_at,
                grace=ENDPOINT_GRACE_S, fallback_silence=ENDPOINT_SILENCE_S)
            if ended:
                log.info(f"[{tag}] end of speech")
                return _end(None)

            try:
                payload = await asyncio.wait_for(device.voice_queue.get(), POLL_S)
            except asyncio.TimeoutError:
                continue

            if payload is None or isinstance(payload, str):
                if payload == VAD_SENTINEL_TIMEOUT or not speech_seen:
                    log.info(f"[{tag}] the Echo heard no speech")
                    return _end("no_speech")
                log.info(f"[{tag}] end of speech (the Echo's VAD)")
                return _end(None)

            if preroll_remaining > 0:
                preroll_remaining -= 1
                continue
            if listening_since is None:
                listening_since = time.monotonic()
                trace.t_first_frame_ms = trace.elapsed_ms()

            if gate is None:
                frames = [payload]
            else:
                prob = await loop.run_in_executor(None, vad.prob, payload)
                frames = gate.push(payload, prob)
                if gate.open and not speech_seen:
                    speech_seen = True
                    log.info(f"[{tag}] speech gate {gate.summary()}, p={prob:.2f}")
                    em_tasks.spawn(device.beam_lock())

            for frame in frames:
                if _is_speech(frame):
                    last_speech_at = time.monotonic()
                    if first_speech_at is None:
                        first_speech_at = last_speech_at
                if gate is None and not speech_seen and last_speech_at is not None:
                    speech_seen = True
                    em_tasks.spawn(device.beam_lock())
                if denoiser is not None:
                    try:
                        frame = await loop.run_in_executor(None, denoiser.process, frame)
                    except Exception as e:
                        log.warning(f"[{tag}] NS failed mid-turn ({e}) — raw audio")
                        denoiser = None
                if recording is not None and len(recording) < em_recordings.MAX_UTTERANCE_BYTES:
                    recording.extend(frame)
                trace.audio_frames += 1
                audio.extend(frame)
    finally:
        if gate is not None and not gate.open:
            log.info(f"[{tag}] speech gate {gate.summary()}")
        if recording:
            device.last_utterance_pcm = bytes(recording)


# ─── 4. Reply ────────────────────────────────────────────────────────────────

def to_wire(audio: em_wyoming.Audio) -> bytes:
    """Wyoming audio → the speaker wire format: mono, 48 kHz, S16."""
    if audio.width != 2:
        raise em_wyoming.WyomingError(f"unsupported sample width {audio.width}")
    samples = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32)
    if audio.channels > 1:
        samples = samples[: len(samples) // audio.channels * audio.channels]
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    if audio.rate != WIRE_RATE and samples.size:
        from scipy.signal import resample_poly
        g = math.gcd(WIRE_RATE, audio.rate)
        samples = resample_poly(samples, WIRE_RATE // g, audio.rate // g)
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


async def _speak(sentence: str) -> bytes:
    audio = await em_wyoming.synthesize(TTS_URI, sentence, TTS_VOICE,
                                        timeout=TTS_TIMEOUT)
    return await asyncio.to_thread(to_wire, audio)


def _chunks(pcm: bytes):
    for i in range(0, len(pcm), WIRE_CHUNK):
        yield pcm[i:i + WIRE_CHUNK]


async def _reply_audio(text: str, history: list[dict], spoken: list[str],
                       trace, tag: str) -> AsyncIterator[bytes]:
    """The reply as wire audio, a sentence at a time. Each sentence is
    appended to `spoken` once it has been synthesized."""
    messages = ([{"role": "system", "content": em_llm.system_prompt(LLM_PROMPT)}]
                + history + [{"role": "user", "content": text}])
    deltas = em_llm.stream_reply(LLM_URL, LLM_MODEL, messages, api_key=LLM_KEY,
                                 max_tokens=LLM_MAX_TOKENS, timeout=REPLY_WAIT)
    try:
        async with contextlib.aclosing(em_llm.sentences(deltas)) as stream:
            async for sentence in stream:
                if not spoken:
                    trace.t_tts_url_ms = trace.elapsed_ms()   # the reply began
                pcm = await _speak(sentence)
                if not spoken:
                    trace.t_tts_fetched_ms = trace.elapsed_ms()
                    log.info(f"[{tag}] replying: {sentence}")
                spoken.append(sentence)
                for chunk in _chunks(pcm):
                    yield chunk
    except em_llm.LLMError as e:
        log.error(f"[{tag}] the model failed: {e}")
        trace.outcome = "llm_error"
        if not spoken:
            for chunk in _chunks(await _speak(APOLOGY_NO_MODEL)):
                yield chunk


async def _play_line(device, trace, post_turn_play, line: str, tag: str) -> None:
    """Say one fixed line, when the turn cannot go on; quietly skipped if
    text to speech is what failed."""
    try:
        pcm = await _speak(line)
    except em_wyoming.WyomingError as e:
        log.error(f"[{tag}] could not say why: {e}")
        return

    async def _one():
        for chunk in _chunks(pcm):
            yield chunk
    async with contextlib.aclosing(_one()) as chunks:
        await post_turn_play(chunks)


def _remember(history: list[dict], text: str, reply: str) -> None:
    if not reply:
        return
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    del history[:-HISTORY_MESSAGES]


# ─── 5. Record ───────────────────────────────────────────────────────────────

async def _persist(device, trace) -> None:
    """The turn's trace, stored and pushed as em_esphome stores its own."""
    trace.t_complete_ms = trace.elapsed_ms()
    if not trace.outcome:
        trace.outcome = "ok"
    trace.emit()
    wi = trace.wake_info or {}
    turn_record = {
        "ts":             time.time(),
        "trigger":        trace.trigger,
        "wake_model":     wi.get("model"),
        "wake_score":     wi.get("score"),
        "wake_threshold": wi.get("threshold"),
        "noise_floor":    wi.get("noise_floor"),
        "outcome":        trace.outcome,
        "total_ms":       trace.t_complete_ms,
        "vad_start_ms":   trace.t_vad_start_ms,
        "vad_end_ms":     trace.t_vad_end_ms,
        "stt_ms":         trace.t_stt_ms,
        "tts_url_ms":     trace.t_tts_url_ms,
        "tts_fetch_ms":   trace.t_tts_fetched_ms,
        "playback_ms":    trace.t_playback_ms,
        "audio_ms":       trace.audio_frames * 80,
        "tts_bytes":      trace.tts_bytes,
        "stt_text":       trace.stt_text,
    }
    await em_turn_record.persist_turn(device, turn_record)
    device.turn_history.append(turn_record)
    with contextlib.suppress(Exception):
        await em_events.push({"type": "turn_complete",
                              "device_id": device.device_id,
                              "turn": turn_record})


# ─── Home Assistant entities: nobody to tell ─────────────────────────────────

def send_button_event(device_id: str, event_type: str) -> None:
    log.info(f"[{device_id}] action button {event_type} (no Home Assistant "
             f"under the AI backend — not forwarded)")


def _nobody_to_tell(*_args, **_kwargs) -> None:
    """A reading or state em_esphome pushes to an HA entity."""
    return None


async def _nobody_to_tell_async(*_args, **_kwargs) -> None:
    return None


# Bound, not defined: tests/_sources.find_def expects each function name to
# be defined once in the controller, and these are em_esphome's.
update_ambient_lux = _nobody_to_tell          # (device_id, lux)
update_temperatures = _nobody_to_tell         # (device_id, cpu_c, max_c)
push_media_state = _nobody_to_tell_async      # (device_id, state)
update_oww_model = _nobody_to_tell_async      # (device_id, model_id)


def update_device_volume(device_id: str, volume: float) -> None:
    _state(device_id).volume = volume


# ─── Picker and mute ─────────────────────────────────────────────────────────

def get_mic_muted_and_wake_word(device_id: str) -> tuple[bool, bool]:
    """(mic muted, wake word on); (False, True) for a device never seen."""
    s = _devices.get(device_id)
    if s is None:
        return False, True
    return s.mic_muted, s.wake_word_enabled


def update_mic_muted(device_id: str, muted: bool) -> None:
    _state(device_id).mic_muted = bool(muted)


def update_wake_word(device_id: str, enabled: bool, refresh: bool = False) -> None:
    _state(device_id).wake_word_enabled = bool(enabled)


# ─── Output mute: HA's media-player mute does not exist here ─────────────────

def output_mute_report(device_id: str, level: int) -> tuple[bool, int | None]:
    return True, None


def output_mute_on_reconnect(device_id: str) -> int | None:
    return None


# ─── Timers: none yet ────────────────────────────────────────────────────────

def clear_timers(device_id: str) -> None:
    return None


async def dismiss_timer_alarm(device_id: str) -> bool:
    return False
