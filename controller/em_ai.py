"""
em_ai.py — the controller answering its Echoes itself, with no Home Assistant.

Selected by VOICE_BACKEND=ai (em_voice.py). Exports the same names as
em_esphome, the Home Assistant backend, so em_controller and the route modules
drive either one through the same calls; tests/test_voice_backend.py holds the
two to that.

What each part of em_esphome's surface means here:

  Voice turns        trigger_voice_turn, cancel_voice_turn, can_serve_turn,
                     abort_ha_run, note_barged — the turn itself. Until the
                     pipeline lands, no turn can be served: can_serve_turn is
                     False, so an Echo stands down before it claims a wake and
                     the wake is recorded as dropped, exactly as with no Home
                     Assistant connected.
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
  Timers             clear_timers, dismiss_timer_alarm — no timers yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import em_turn_record
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


_devices: dict[str, _DeviceState] = {}


def _state(device_id: str) -> _DeviceState:
    return _devices.setdefault(device_id, _DeviceState())


# ─── Lifecycle ───────────────────────────────────────────────────────────────

async def start_esphome_servers(devices: dict, host: str = "0.0.0.0") -> None:
    """Nothing to listen on: the AI backend has no satellites for HA to dial."""
    log.info("Voice backend: AI (VOICE_BACKEND=ai) — no Home Assistant satellites")


async def stop_esphome_servers() -> None:
    _devices.clear()


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
    """Whether a turn started now could be answered. Not yet: the pipeline is
    still to come, and an Echo that cannot answer must stand down before it
    claims a wake, so one that could is not silenced by it."""
    return False


async def record_dropped_wake(device, trigger_label: str, wake_info) -> None:
    await em_turn_record.record_dropped_turn(device, trigger_label, wake_info)


async def trigger_voice_turn(
    device,
    on_thinking,
    post_turn_play,
    trigger_label: str = "unknown",
    preroll_discard: int = VOICE_PREROLL_DISCARD,
) -> bool:
    """Run one voice turn; return whether the conversation continues.
    Until the pipeline lands every turn is recorded as dropped."""
    wake_info = device.last_wake
    device.last_wake = None
    log.warning(f"[{device.device_id}] AI backend: voice turns are not "
                f"implemented yet — turn dropped")
    await em_turn_record.record_dropped_turn(device, trigger_label, wake_info)
    return False


def cancel_voice_turn(device_id: str, abort_ha: bool = False,
                      reason: str = "cancelled") -> None:
    return None


def abort_ha_run(device_id: str) -> None:
    return None


def note_barged(device_id: str) -> None:
    return None


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
