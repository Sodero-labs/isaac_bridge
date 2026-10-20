"""The moderator's per-step exchange and the inbound queue, against fake devices and views."""
from __future__ import annotations

import logging
import types

import numpy as np

from fakes import FakeHandler, command, make_moderator, make_robot
from scripts import protocol
from scripts.moderator import interface
from scripts.moderator import moderator as moderator_mod


class _Device:
    """A device that records what the moderator does with it."""

    def __init__(self, path="/World/r", log=None):
        self._path = path
        self.log = [] if log is None else log
        self.targets = []

    def synchronize(self):
        self.log.append("device")

    def get_actual(self):
        return [("sim_state", b"state")]

    def update_target(self, data_type, payload):
        self.targets.append((data_type, payload))

    def latch_command(self):
        self.log.append("latch")


# ------------------------------------------------------------------ inbound queue
def _handler():
    return interface.MessageHandler(host="127.0.0.1", subscribe_port=1, publish_port=2)


def test_drain_inbound_hands_frames_over_in_order():
    h = _handler()
    seen = []
    h.inbound.append([b"TM/", b"r", b"sim_command", b"1"])
    h.inbound.append([b"TM/", b"r", b"sim_command", b"2"])
    assert h.drain_inbound(lambda d: seen.append(d[3])) == 2
    assert seen == [b"1", b"2"] and not h.inbound


def test_drain_inbound_survives_a_failing_handler():
    h = _handler()
    h.inbound.extend([[b"TM/", b"r", b"x", b"bad"], [b"TM/", b"r", b"x", b"ok"]])
    seen = []

    def handle(d):
        if d[3] == b"bad":
            raise RuntimeError("boom")
        seen.append(d[3])

    assert h.drain_inbound(handle) == 2 and seen == [b"ok"]


def test_inbound_queue_is_bounded():
    h = _handler()
    for i in range(interface.INBOUND_MAXLEN + 50):
        h.inbound.append([b"TM/", b"r", b"x", str(i).encode()])
    assert len(h.inbound) == interface.INBOUND_MAXLEN and h.inbound[0][3] == b"50"


def test_a_stopped_handler_publishes_nothing():
    h = _handler()
    h.send("r", [("sim_state", b"x")])           # no socket, not running: silently nothing


# ------------------------------------------------------------------ per step
def test_each_step_drains_the_queue_and_samples_the_link_view():
    sampled = []
    handler = FakeHandler()
    device = _Device()
    m = make_moderator({"r": device}, handler=handler,
                       link_view=types.SimpleNamespace(sample=lambda step: sampled.append(step) or True))
    handler.inbound.append([b"TM/", b"r", b"sim_command", b"x"])
    m.synchronize(1 / 250)
    assert device.targets == [("sim_command", b"x")] and sampled == [1]
    m.synchronize(1 / 250)
    assert sampled == [1, 2]
    assert handler.sent == [("r", [("sim_state", b"state")])] * 2


def test_the_batch_is_sampled_before_and_flushed_after_the_devices():
    order = []
    batch = types.SimpleNamespace(sample=lambda s: order.append("sample") or True, flush=lambda: order.append("flush"))
    m = make_moderator({"r": _Device(log=order)}, batch=batch)
    m.synchronize(1 / 250)
    assert order == ["sample", "device", "flush"]


def test_the_batch_waits_for_every_articulation_to_be_bound():
    sampled, flushed = [], []
    batch = types.SimpleNamespace(sample=lambda s: sampled.append(s) or True, flush=lambda: flushed.append(1))
    device_map = {moderator_mod.ARTICULATION: {"go1": {"path": "/World/go1"}, "duck": {"path": "/World/duck"}},
                  moderator_mod.RIGID_OBJECT: {}}
    m = make_moderator({}, device_map=device_map, batch=batch)
    m._retry_tick = 5                                # no bind attempts in this test
    assert m._articulations_ready() is False
    m.synchronize(1 / 250)
    assert sampled == [] and flushed == []           # no stage-wide view while the robots are not bound
    m._instance_map = {"go1": _Device("/World/go1")}
    m.synchronize(1 / 250)
    assert sampled == []                             # one robot still missing
    m._instance_map["duck"] = _Device("/World/duck")
    m.synchronize(1 / 250)
    assert sampled == [3] and flushed == [1]


def test_articulations_ready_needs_at_least_one_articulation():
    device_map = {moderator_mod.ARTICULATION: {}, moderator_mod.RIGID_OBJECT: {"ball": {"path": "/W/b"}}}
    assert make_moderator({}, device_map=device_map)._articulations_ready() is False


def test_articulation_paths_and_view_attach():
    attached = []

    class _Dev:
        _path = "/World/go1"

        def set_link_view(self, v):
            attached.append(("link", v))

        def set_batch(self, b):
            attached.append(("batch", b))

    device_map = {moderator_mod.ARTICULATION: {"go1": {"path": "/World/go1"}},
                  moderator_mod.RIGID_OBJECT: {"ball": {"path": "/World/ball"}}}
    m = make_moderator({"go1": _Dev()}, device_map=device_map)
    assert m.articulation_paths() == ["/World/go1"]
    sentinel = object()
    m.set_batch(sentinel)
    assert ("batch", sentinel) in attached


def test_states_are_muted_for_a_while_after_a_bind():
    handler = FakeHandler()
    m = make_moderator({"r": _Device()}, handler=handler)
    m._mute_steps = 3
    for _ in range(5):
        m.synchronize(1 / 250)
    assert len(handler.sent) == 2


# ------------------------------------------------------------------ frame latch
def test_frame_latch_is_off_by_default():
    log = []
    m = make_moderator({"r": _Device(log=log)})
    m.on_frame()
    m.synchronize(1 / 250)
    assert "latch" not in log


def test_frame_latch_snapshots_at_the_first_step_of_each_frame(monkeypatch):
    monkeypatch.setenv("RC_ISAAC_FRAME_LATCH", "1")
    log = []
    m = make_moderator({"r": _Device(log=log)})
    m.on_frame()
    for _ in range(4):                               # one rendered frame, four physics steps
        m.synchronize(1 / 240)
    m.on_frame()
    m.synchronize(1 / 240)
    assert log.count("latch") == 2


def test_a_robot_without_the_latch_keeps_driving_on_change_only():
    a, pv = make_robot()
    a.update_target("sim_command", command(3, [0.1, 0.2, 0.3]))
    a.synchronize()
    pv.calls.clear()
    a.latch_command()                                # the moderator's frame tick, latch off
    a.synchronize()
    assert pv.writes() == [] and a._latched is None


# ------------------------------------------------------------------ binding and routing
def test_a_device_that_cannot_bind_is_logged_once_and_retried(caplog):
    device_map = {moderator_mod.ARTICULATION: {"go1": {"path": "/World/go1"}}, moderator_mod.RIGID_OBJECT: {}}
    m = make_moderator({}, device_map=device_map)
    attempts = []
    m._class_map[moderator_mod.ARTICULATION] = lambda **kw: attempts.append(kw["name"]) or None
    with caplog.at_level(logging.INFO, logger="isaac_bridge"):
        for _ in range(2 * (m.RETRY_EVERY_STEPS + 1) + 1):
            m.synchronize(1 / 200)
    assert attempts == ["go1"] * 3                   # the first step, then every RETRY_EVERY_STEPS + 1 steps
    assert caplog.text.count("physics not ready yet") == 1


def test_frames_for_an_unknown_robot_are_warned_about_once(caplog):
    device = _Device("/World/go1")
    m = make_moderator({"go1": device})
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        for _ in range(100):
            m._process_data([b"TM/", b"go1_typo", b"sim_command", b"x"])
            m._process_data([b"TM/", b"go1", b"sim_command", b"y"])
    warnings = [r.getMessage() for r in caplog.records if "go1_typo" in r.getMessage()]
    assert len(warnings) == 1 and "robots: ['go1']" in warnings[0]
    assert len(device.targets) == 100


def test_frames_for_a_robot_not_bound_yet_are_not_a_warning(caplog):
    device_map = {moderator_mod.ARTICULATION: {"go1": {"path": "/World/go1"}}, moderator_mod.RIGID_OBJECT: {}}
    m = make_moderator({}, device_map=device_map)
    with caplog.at_level(logging.INFO, logger="isaac_bridge"):
        for _ in range(10):
            m._process_data([b"TM/", b"go1", b"sim_command", b"x"])
    assert [r.levelno for r in caplog.records if "go1" in r.getMessage()] == [logging.INFO]


# ------------------------------------------------------------------ sync errors
class _Broken:
    _path = "/World/duck"

    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1
        raise ZeroDivisionError("float division by zero")

    def get_actual(self):
        return []

    def destroy(self):
        pass


def test_sync_errors_are_warnings_rate_limited_per_device(caplog):
    m = make_moderator({})
    exc = ZeroDivisionError("float division by zero")
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        m._report_sync_error("duck", exc, now=100.0)
        for i in range(1, 2000):                     # 10 s of 200 Hz steps
            m._report_sync_error("duck", exc, now=100.0 + i * 0.005)
        m._report_sync_error("go1", exc, now=100.5)   # another robot reports on its own
        m._report_sync_error("duck", exc, now=110.0)
    duck = [r for r in caplog.records if "sync error in duck" in r.getMessage()]
    assert len(duck) == 2 and "2000 error(s) in the last 10 s" in duck[1].getMessage()
    assert duck[0].exc_info is not None                              # the traceback once
    assert any("sync error in go1" in r.getMessage() for r in caplog.records)


def test_a_raising_device_is_reported_and_forgotten_with_its_instance(caplog):
    broken = _Broken()
    m = make_moderator({"duck": broken})
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        for _ in range(5):
            m.synchronize(1 / 200)
    assert broken.calls == 5
    assert len([r for r in caplog.records if "sync error in duck" in r.getMessage()]) == 1
    m._reset_instance_map()
    assert m._sync_errors == {}


# ------------------------------------------------------------------ IMU and contact passes
class _ImuDevice:
    def __init__(self, path, imu=True):
        self._path = path
        self._imu = imu
        self.batched = None
        self.blobs = 0

    def has_imu(self):
        return bool(self._imu)

    def synchronize(self):
        pass

    def get_actual(self):
        return []

    def set_imu_batched(self, on):
        self.batched = on

    def set_imu_blob(self, blob, seq):
        self.blobs += 1

    def imu_binding(self):
        return None if not self._imu else {
            "path": f"{self._path}/links/base", "contact_paths": [], "offset": np.zeros(3),
            "contact_offsets": [], "contact_height": 0.01, "ground_z": 0.0, "accel_tau": 0.02}


def test_the_imu_pass_is_built_once_and_rebuilt_when_robots_appear():
    m = make_moderator({"a": _ImuDevice("/World/a")}, sim_time=1.0)
    m.synchronize(1 / 250)
    first = m._imu_batch
    assert first is not None and len(first.devices) == 1
    m.synchronize(1 / 250)
    assert m._imu_batch is first                         # unchanged device set: no rebuild
    m._instance_map["b"] = _ImuDevice("/World/b")
    m.synchronize(1 / 250)
    assert m._imu_batch is not first and len(m._imu_batch.devices) == 2


def test_devices_learn_whether_the_imu_pass_computed_them():
    d = _ImuDevice("/World/a")
    m = make_moderator({"a": d})
    m.synchronize(1 / 250)
    assert d.batched is False and d.blobs == 0           # no link view: the per-robot path this step


def test_devices_without_an_imu_stay_out_of_the_pass():
    m = make_moderator({"a": _ImuDevice("/World/a", imu=False), "b": _ImuDevice("/World/b")})
    m.synchronize(1 / 250)
    assert [d._path for d in m._imu_batch.devices] == ["/World/b"]


def test_a_device_dropped_from_a_rebuilt_pass_stops_relying_on_it():
    keep, lost = _ImuDevice("/World/a"), _ImuDevice("/World/b")
    m = make_moderator({"a": keep, "b": lost})
    m.synchronize(1 / 250)
    assert len(m._imu_batch.devices) == 2
    lost.batched = True                                  # the old pass had claimed it
    lost._imu = False                                    # ... and now it has no IMU binding
    m._imu_batch_dirty = True
    m.synchronize(1 / 250)
    assert [d._path for d in m._imu_batch.devices] == ["/World/a"]
    assert lost.batched is False


def test_a_published_state_goes_out_through_the_message_handler():
    sent = []
    h = interface.MessageHandler(host="127.0.0.1", subscribe_port=1, publish_port=2)
    h._is_server_running = True
    h._publish_socket = types.SimpleNamespace(send=lambda frame, flags=0: sent.append((frame, flags)))
    a, _ = make_robot(path="/World/go1")
    m = make_moderator({"go1": a}, handler=h)
    m.synchronize(1 / 250)
    assert [frame.split(b"||", 1)[0] for frame, _ in sent] == [b"TM/go1/sim_state"]
    assert all(flags == interface.zmq.NOBLOCK for _, flags in sent)
    assert len(sent[0][0].split(b"||", 1)[1]) == protocol.SIM_STATE_SIZE
