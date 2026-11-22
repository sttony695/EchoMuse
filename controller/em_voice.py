"""
em_voice.py — which voice backend answers this controller's Echoes.

  VOICE_BACKEND=esphome   (default) em_esphome: every Echo is an ESPHome voice
                          satellite and Home Assistant's Assist answers it.
  VOICE_BACKEND=ai        em_ai: the controller answers itself — speech to
                          text, an LLM and text to speech, with no Home
                          Assistant.

Modules that used `import em_esphome` import this under the same name
(`import em_voice as em_esphome`, `import em_voice as esphome`), so every call
site, and every test that reads those call sites as source, is unchanged.

Each name is bound here explicitly, not forwarded by a module __getattr__:
tests/test_module_attrs.py reads every `module.name` statically and must find
it defined. A test that patches `em_controller.esphome.x` patches this module,
which is what em_controller reads. tests/test_voice_backend.py holds this list
to the call sites and both backends to it. Read once at import: a change
needs a restart.
"""

from __future__ import annotations

import os

BACKENDS = ("esphome", "ai")


def choose(value: str | None) -> str:
    """The backend VOICE_BACKEND names. Unset or blank is esphome; anything
    else unrecognised refuses to start, rather than quietly running the
    other backend and leaving every Echo without the one asked for."""
    name = (value or "").strip().lower() or "esphome"
    if name not in BACKENDS:
        raise SystemExit(
            f"VOICE_BACKEND={value!r} is not one of {', '.join(BACKENDS)}")
    return name


BACKEND = choose(os.environ.get("VOICE_BACKEND"))

if BACKEND == "ai":
    import em_ai as backend
else:
    import em_esphome as backend

# Constants shared by both backends.
BUTTON_HOLD_MS = backend.BUTTON_HOLD_MS
CONVERSATION_TRIGGER = backend.CONVERSATION_TRIGGER
VAD_SENTINEL_END = backend.VAD_SENTINEL_END
VAD_SENTINEL_TIMEOUT = backend.VAD_SENTINEL_TIMEOUT
VOICE_PREROLL_DISCARD = backend.VOICE_PREROLL_DISCARD

# Lifecycle.
start_esphome_servers = backend.start_esphome_servers
stop_esphome_servers = backend.stop_esphome_servers
device_connected = backend.device_connected
device_disconnected = backend.device_disconnected
device_deleted = backend.device_deleted
set_device_capabilities = backend.set_device_capabilities
get_status = backend.get_status

# Voice turns.
can_serve_turn = backend.can_serve_turn
record_dropped_wake = backend.record_dropped_wake
trigger_voice_turn = backend.trigger_voice_turn
cancel_voice_turn = backend.cancel_voice_turn
abort_ha_run = backend.abort_ha_run
note_barged = backend.note_barged

# What em_esphome tells Home Assistant.
send_button_event = backend.send_button_event
update_ambient_lux = backend.update_ambient_lux
update_temperatures = backend.update_temperatures
push_media_state = backend.push_media_state
update_oww_model = backend.update_oww_model
update_device_volume = backend.update_device_volume

# Picker, mute and output mute.
get_mic_muted_and_wake_word = backend.get_mic_muted_and_wake_word
update_mic_muted = backend.update_mic_muted
update_wake_word = backend.update_wake_word
output_mute_report = backend.output_mute_report
output_mute_on_reconnect = backend.output_mute_on_reconnect

# Timers.
clear_timers = backend.clear_timers
dismiss_timer_alarm = backend.dismiss_timer_alarm
