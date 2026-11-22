"""
em_voice picks the voice backend; em_ai must answer every call em_esphome does.

The call sites are read from source, the way test_layering reads imports, so a
new `esphome.foo(...)` in em_controller, or `em_esphome.foo` in a route
module, fails here until em_ai has `foo` too — the alternative is an
AttributeError the first time an Echo does that thing under VOICE_BACKEND=ai.
"""

import ast
import asyncio
import importlib
import inspect
import sys
import types

import pytest

import _sources

# Each importer and the name it binds em_voice to.
IMPORTERS = {
    "em_controller": "esphome",
    "em_device_config": "em_esphome",
    "em_device_view": "em_esphome",
    "em_oww_sync": "em_esphome",
    "em_routes_devices": "em_esphome",
}


def _attributes_used(module: str, alias: str) -> set[str]:
    tree = _sources.tree(next(m for m in _sources.MODULES if m.stem == module))
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == alias}


def _imports_em_voice_as(module: str) -> set[str]:
    tree = _sources.tree(next(m for m in _sources.MODULES if m.stem == module))
    return {a.asname or a.name for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for a in node.names if a.name == "em_voice"}


def test_every_importer_goes_through_em_voice():
    for module, alias in IMPORTERS.items():
        assert _imports_em_voice_as(module) == {alias}, module


def test_nothing_else_imports_em_esphome_by_name():
    # em_ble_proxy reaches into em_esphome for the satellite's zeroconf and
    # identity helpers, which exist only with that backend and are guarded
    # (None-checked) there; em_ai re-exports em_esphome's shared constants;
    # em_voice is the switch.
    allowed = {"em_ble_proxy", "em_ai", "em_voice"}
    found = set()
    for path in _sources.MODULES:
        for node in ast.walk(_sources.tree(path)):
            if isinstance(node, ast.Import) and any(
                    a.name == "em_esphome" for a in node.names):
                found.add(path.stem)
            elif isinstance(node, ast.ImportFrom) and node.module == "em_esphome":
                found.add(path.stem)
    assert found <= allowed, sorted(found - allowed)


def test_em_ai_answers_every_call_site():
    import em_ai
    used = set()
    for module, alias in IMPORTERS.items():
        used |= _attributes_used(module, alias)
    assert len(used) > 20                       # the scan found the calls
    missing = sorted(n for n in used if not hasattr(em_ai, n))
    assert not missing, f"em_ai lacks {missing}"


def test_em_voice_binds_every_call_site():
    import em_voice
    used = set()
    for module, alias in IMPORTERS.items():
        used |= _attributes_used(module, alias)
    missing = sorted(n for n in used if n not in vars(em_voice))
    assert not missing, f"em_voice does not bind {missing}"


def test_em_ai_signatures_match_em_esphome():
    import em_ai
    import em_esphome
    used = set()
    for module, alias in IMPORTERS.items():
        used |= _attributes_used(module, alias)
    no_ops = (em_ai._nobody_to_tell, em_ai._nobody_to_tell_async)
    for name in sorted(used):
        theirs, ours = getattr(em_esphome, name), getattr(em_ai, name)
        if not callable(theirs):
            assert ours == theirs, name         # shared constants, one value
            continue
        assert inspect.iscoroutinefunction(ours) == \
            inspect.iscoroutinefunction(theirs), f"{name}: async differs"
        if ours in no_ops:
            continue                            # takes anything, does nothing
        assert list(inspect.signature(ours).parameters) == \
            list(inspect.signature(theirs).parameters), f"{name}: parameters differ"


@pytest.mark.parametrize("value, want", [
    (None, "esphome"), ("", "esphome"), ("  ", "esphome"),
    ("esphome", "esphome"), ("AI", "ai"), (" ai ", "ai"),
])
def test_choose(value, want):
    import em_voice
    assert em_voice.choose(value) == want


def test_an_unknown_backend_refuses_to_start():
    import em_voice
    with pytest.raises(SystemExit):
        em_voice.choose("assist")


def _fresh_em_voice(monkeypatch, value):
    monkeypatch.setenv("VOICE_BACKEND", value)
    monkeypatch.delitem(sys.modules, "em_voice", raising=False)
    module = importlib.import_module("em_voice")
    monkeypatch.delitem(sys.modules, "em_voice", raising=False)
    return module


def test_the_default_forwards_to_em_esphome(monkeypatch):
    import em_esphome
    voice = _fresh_em_voice(monkeypatch, "")
    assert voice.BACKEND == "esphome"
    assert voice.can_serve_turn is em_esphome.can_serve_turn


def test_ai_forwards_to_em_ai(monkeypatch):
    import em_ai
    voice = _fresh_em_voice(monkeypatch, "ai")
    assert voice.BACKEND == "ai"
    assert voice.trigger_voice_turn is em_ai.trigger_voice_turn


# ── the AI backend before its pipeline exists ───────────────────────────────

def test_ai_serves_no_turn_yet():
    import em_ai
    assert em_ai.can_serve_turn("G090LF0000000001") is False


def test_ai_turn_is_recorded_as_dropped(monkeypatch):
    import em_ai
    dropped = []

    async def record(device, label, wake_info):
        dropped.append((device.device_id, label, wake_info))

    monkeypatch.setattr(em_ai.em_turn_record, "record_dropped_turn", record)
    device = types.SimpleNamespace(device_id="G090LF0000000001",
                                   last_wake={"score": 0.9})

    async def no_op(*_a, **_k):
        return None

    went_on = asyncio.run(em_ai.trigger_voice_turn(
        device, no_op, no_op, trigger_label="wakeword(0.9)"))
    assert went_on is False
    assert dropped == [("G090LF0000000001", "wakeword(0.9)", {"score": 0.9})]
    assert device.last_wake is None             # popped, as em_esphome does


def test_ai_remembers_mute_and_wake_word():
    import em_ai
    dev = "G090LF0000000002"
    assert em_ai.get_mic_muted_and_wake_word(dev) == (False, True)
    em_ai.update_mic_muted(dev, True)
    em_ai.update_wake_word(dev, False)
    assert em_ai.get_mic_muted_and_wake_word(dev) == (True, False)
    asyncio.run(em_ai.device_deleted(dev))
    assert em_ai.get_mic_muted_and_wake_word(dev) == (False, True)


def test_ai_keeps_every_volume_report():
    import em_ai
    assert em_ai.output_mute_report("G090LF0000000003", 0) == (True, None)
    assert em_ai.output_mute_on_reconnect("G090LF0000000003") is None


def test_ai_connect_and_disconnect():
    import em_ai
    dev = "G090LF0000000004"

    async def go():
        await em_ai.device_connected(dev, wake_word_enabled=False)
        assert em_ai._devices[dev].connected
        assert em_ai.get_mic_muted_and_wake_word(dev) == (False, False)
        await em_ai.device_disconnected(dev)
        assert not em_ai._devices[dev].connected
        await em_ai.stop_esphome_servers()
        assert dev not in em_ai._devices
    asyncio.run(go())
