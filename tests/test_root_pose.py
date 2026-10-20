"""root_pose frames: every articulation's PhysX root link world pose on the state port, at most 20 Hz.

The payload encoding, the sim-time rate limit, where the pose is read from (the link view first, the articulation
view as the fallback), and that a robot without pose data this step publishes its other frames and never raises."""
from __future__ import annotations

import json
import logging
import types

import numpy as np
import pytest

from fakes import FakeSimView, make_moderator, make_robot, make_tracked_object, sampled_link_view
from scripts import config
from scripts.articulation_batch import ArticulationBatch
from scripts.device import articulation, root_pose
from scripts.link_view import LinkStateView
from scripts.moderator import interface

ROOT = "/W/r/links/base"
LINKS = ["/W/r/links/thigh", ROOT, "/W/r/links/foot"]        # PhysX order need not start at the root


def _xf(pos, quat_wxyz):
    """Tensor layout of a transform: x y z qx qy qz qw."""
    w, x, y, z = quat_wxyz
    return np.array([*pos, x, y, z, w], dtype=np.float64)


def _world_store():
    s = 0.5 ** 0.5
    return {
        "/W/r/links/thigh": (_xf([9.0, 9.0, 9.0], [1, 0, 0, 0]), np.full(6, 9.0)),
        ROOT: (_xf([1.25, -0.5, 0.31], [s, 0.0, 0.0, s]), np.array([0.4, 0.1, 0.0, 0.0, 0.0, 0.8])),
        "/W/r/links/foot": (_xf([8.0, 8.0, 8.0], [1, 0, 0, 0]), np.full(6, 8.0)),
        "/W/ball": (_xf([3.0, 3.0, 0.1], [1, 0, 0, 0]), np.zeros(6)),
    }


class _Meta:
    def __init__(self, parents):
        self.link_parents = parents


def _robot(*, link_view=None, links=LINKS, parents=("base", "", "thigh"), root_xf=None, root_vel=None):
    """A robot on the raw-view fakes whose view names its links (the root one parentless)."""
    a, pv = make_robot(3)
    pv.link_paths = [list(links)] if links is not None else None
    pv.shared_metatype = _Meta(list(parents))
    if root_xf is not None:
        pv.get_root_transforms = lambda: np.asarray(root_xf, dtype=np.float32).reshape(1, 7)
        pv.get_root_velocities = lambda: np.asarray(root_vel, dtype=np.float32).reshape(1, 6)
    a.set_link_view(link_view)
    a._sim_context.current_time = 0.0
    return a, pv


def _root_frames(frames):
    return [json.loads(p) for t, p in frames if t == root_pose.ROOT_POSE_TOPIC]


# ------------------------------------------------------------------------------------------ encoding
def test_encode_is_compact_json_with_the_documented_fields():
    payload = root_pose.encode_root_pose(12.3456789, 7, [1, 2, 3], [1, 0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 1.5])
    assert isinstance(payload, bytes) and b" " not in payload
    msg = json.loads(payload)
    assert msg == {"pos": [1.0, 2.0, 3.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "lin_vel": [0.1, 0.2, 0.3],
                   "ang_vel": [0.0, 0.0, 1.5], "sim_time": 12.345679, "seq": 7}
    assert root_pose.decode_root_pose(payload) == msg


def test_encode_rounds_like_object_state_and_accepts_float32():
    msg = json.loads(root_pose.encode_root_pose(
        np.float32(0.5), 1, np.array([1.123456789, 0, 0], np.float32), np.array([0.70710678, 0, 0, 0.70710678]),
        [1.23456, 0, 0], [0, 0, 0.98765]))
    assert msg["pos"][0] == pytest.approx(1.12346, abs=1e-5)
    assert msg["quat_wxyz"] == [0.707107, 0.0, 0.0, 0.707107]
    assert msg["lin_vel"][0] == 1.2346 and msg["ang_vel"][2] == 0.9877 and msg["sim_time"] == 0.5


@pytest.mark.parametrize("args", [
    (0.1, 1, [np.nan, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]),
    (0.1, 1, [0, 0, 0], [1, 0, 0, 0], [np.inf, 0, 0], [0, 0, 0]),
    (float("nan"), 1, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]),
    (0.1, 1, [0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]),                 # wrong sizes
    (0.1, 1, [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]),
    (0.1, 1, None, [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]),                   # missing
    (0.1, 1, ["a", 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]),
])
def test_encode_refuses_non_finite_or_malformed_values(args):
    assert root_pose.encode_root_pose(*args) is None


@pytest.mark.parametrize("payload", [
    b"not json", b"\xff\xfe", b"[1, 2]", b'{"pos": [0, 0, 0]}',
    b'{"pos": [0, 0], "quat_wxyz": [1, 0, 0, 0], "lin_vel": [0, 0, 0], "ang_vel": [0, 0, 0], "sim_time": 0}',
    b'{"pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0], "lin_vel": [0, 0, 0], "ang_vel": [0, 0, 0], "sim_time": "x"}',
])
def test_decode_rejects_what_is_not_a_root_pose(payload):
    with pytest.raises(ValueError):
        root_pose.decode_root_pose(payload)


# ------------------------------------------------------------------------------------------ rate
@pytest.mark.parametrize("raw, hz", [(None, 20.0), ("", 20.0), ("5", 5.0), ("2.5", 2.5), ("50", 20.0),
                                     ("0", 0.0), ("-3", 0.0), ("nan", 0.0), ("fast", 20.0)])
def test_rate_env_is_clamped_to_20_hz_and_zero_turns_it_off(raw, hz):
    env = {} if raw is None else {config.ROOT_POSE_HZ_ENV: raw}
    assert root_pose.root_pose_hz_from_env(env) == hz


def test_rate_limit_holds_20_hz_of_sim_time_at_250_hz_physics():
    rate = root_pose.RootPoseRate()
    published = []
    for step in range(250):                        # one second of a 250 Hz world
        t = step * 0.004
        if rate.due(t):
            rate.mark(t)
            published.append(t)
    assert 19 <= len(published) <= 20
    assert min(np.diff(published)) >= 0.05 - 1e-9


def test_rate_limit_is_only_moved_by_mark_and_publishes_after_a_reset():
    rate = root_pose.RootPoseRate(20.0)
    assert rate.enabled and rate.due(1.0)
    assert rate.due(1.01)                          # nothing published yet: still due
    rate.mark(1.01)
    assert not rate.due(1.03) and rate.due(1.06)
    assert rate.due(0.0)                           # timeline reset sent the clock back: publish at once
    assert root_pose.RootPoseRate(80.0).period == pytest.approx(0.05)     # never faster than 20 Hz
    off = root_pose.RootPoseRate(0.0)
    assert not off.enabled and not off.due(5.0)


# ------------------------------------------------------------------------------------------ articulation
def test_root_pose_comes_from_the_root_link_row_of_the_shared_link_view():
    store = _world_store()
    a, pv = _robot(link_view=sampled_link_view(store))
    a.synchronize()
    frames = a.get_actual()
    assert [t for t, _ in frames] == ["sim_state", "root_pose"]          # appended after every existing frame
    (msg,) = _root_frames(frames)
    assert set(msg) == {"pos", "quat_wxyz", "lin_vel", "ang_vel", "sim_time", "seq"}
    assert msg["pos"] == pytest.approx([1.25, -0.5, 0.31], abs=1e-5)                  # the ROOT link, not link 0
    assert msg["quat_wxyz"] == pytest.approx([0.5 ** 0.5, 0.0, 0.0, 0.5 ** 0.5], abs=1e-6)   # xyzw -> wxyz
    assert msg["lin_vel"] == pytest.approx([0.4, 0.1, 0.0], abs=1e-4)
    assert msg["ang_vel"] == pytest.approx([0.0, 0.0, 0.8], abs=1e-4)
    assert msg["sim_time"] == 0.0 and msg["seq"] == 1
    assert a._root_link == (ROOT, "root link")
    assert all(c in ("get_q", "get_dq") for c in pv.calls)              # no extra backend call for the pose


def test_root_pose_is_rate_limited_while_sim_state_goes_out_every_step():
    store = _world_store()
    lv = sampled_link_view(store)
    a, _ = _robot(link_view=lv)
    kinds, seqs, times = [], [], []
    for step in range(250):
        a._sim_context.current_time = step * 0.004
        lv.sample(step + 2)
        a.synchronize()
        frames = a.get_actual()
        kinds.append([t for t, _ in frames])
        for msg in _root_frames(frames):
            seqs.append(msg["seq"])
            times.append(msg["sim_time"])
    assert all(k[0] == "sim_state" for k in kinds)
    assert 19 <= len(seqs) <= 20 and seqs == list(range(1, len(seqs) + 1))
    assert min(np.diff(times)) >= 0.05 - 1e-6


def test_other_frames_are_byte_identical_with_and_without_root_pose():
    store = _world_store()
    a, _ = _robot(link_view=sampled_link_view(store))
    b, _ = _robot(link_view=sampled_link_view(store))
    b._root_rate = root_pose.RootPoseRate(0.0)                           # RC_ISAAC_ROOT_POSE_HZ=0
    a._imu_payload = b._imu_payload = b"imu"
    a.synchronize()
    b.synchronize()
    fa, fb = a.get_actual(), b.get_actual()
    assert fa[:-1] == fb and fa[-1][0] == "root_pose"


def test_env_zero_turns_root_pose_off(monkeypatch):
    monkeypatch.setenv(config.ROOT_POSE_HZ_ENV, "0")
    a, _ = make_robot()
    assert not a._root_rate.enabled
    monkeypatch.delenv(config.ROOT_POSE_HZ_ENV)
    b, _ = make_robot()
    assert b._root_rate.period == pytest.approx(1.0 / root_pose.ROOT_POSE_PUB_HZ)


# ------------------------------------------------------------------------------------------ absence of pose data
def test_unsampled_link_view_skips_the_frame_and_the_next_step_publishes():
    store = _world_store()
    holder = {"view": None}
    lv = LinkStateView(list(store), sim_view_factory=lambda: holder["view"], logger=logging.getLogger("t"))
    assert lv.sample(1) is False                                          # physics not up yet
    a, pv = _robot(link_view=lv)
    a.synchronize()
    frames = a.get_actual()
    assert [t for t, _ in frames] == ["sim_state"]
    assert a._root_rate.last_t is None                                    # nothing published: still due
    holder["view"] = FakeSimView(store)
    a._sim_context.current_time = 0.004                                   # the very next step, inside 50 ms
    lv.sample(2)
    a.synchronize()
    assert len(_root_frames(a.get_actual())) == 1


def test_no_link_view_and_no_articulation_view_publishes_nothing_and_never_raises():
    a, pv = _robot(link_view=None)
    del pv.link_paths                                                     # a view that exposes nothing
    a.synchronize()
    a._raw_ok = False                                                     # ... and no numpy-backed raw view
    assert [t for t, _ in a.get_actual()] == ["sim_state"]


def test_missing_physics_view_means_no_root_pose_yet():
    store = _world_store()
    a, pv = _robot(link_view=sampled_link_view(store))
    a.synchronize()
    a._art._physics_view = None
    assert [t for t, _ in a.get_actual()] == ["sim_state"] and a._root_link is None
    a._art._physics_view = pv
    assert len(_root_frames(a.get_actual())) == 1


def test_a_link_view_that_raises_only_drops_the_root_pose(caplog):
    store = _world_store()
    lv = sampled_link_view(store)
    a, _ = _robot(link_view=lv)
    a.synchronize()

    def boom(row):
        raise RuntimeError("view gone")

    lv.pose_wxyz = boom
    with caplog.at_level(logging.DEBUG, logger="isaac_bridge"):
        frames = a.get_actual()
    assert [t for t, _ in frames] == ["sim_state"]
    assert "root_pose skipped: RuntimeError" in caplog.text


def test_non_finite_root_pose_is_not_published_and_warned_once(caplog):
    store = _world_store()
    store[ROOT] = (np.full(7, np.nan), np.zeros(6))
    lv = sampled_link_view(store)
    a, _ = _robot(link_view=lv)
    with caplog.at_level(logging.WARNING):
        for step in range(40):
            a._sim_context.current_time = step * 0.004
            lv.sample(step + 2)
            a.synchronize()
            assert "root_pose" not in [t for t, _ in a.get_actual()]
    assert caplog.text.count("root link pose is not finite") == 1
    assert a._root_skips == 4                          # t = 0, 0.052, 0.104, 0.156: checked at <= 20 Hz, not every step


# ------------------------------------------------------------------------------------------ fallback: articulation view
def test_root_link_outside_the_link_view_reads_the_raw_articulation_view():
    store = _world_store()
    lv = sampled_link_view(store, paths=["/W/ball"])                      # root link not tracked by the view
    a, pv = _robot(link_view=lv, root_xf=_xf([2.0, 1.0, 0.5], [1, 0, 0, 0]), root_vel=[1, 0, 0, 0, 0, 0.25])
    a.synchronize()
    (msg,) = _root_frames(a.get_actual())
    assert msg["pos"] == pytest.approx([2.0, 1.0, 0.5]) and msg["quat_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert msg["lin_vel"] == [1.0, 0.0, 0.0] and msg["ang_vel"] == [0.0, 0.0, 0.25]
    assert a._root_row == -1 and a._root_source == "articulation view"


def test_fallback_through_the_world_batch_is_one_backend_call_for_every_robot():
    class _View:
        def __init__(self):
            self.prim_paths = ["/W/r", "/W/s"]
            self.count, self.max_dofs = 2, 3
            self.q = np.zeros((2, 3), np.float32)
            self.root_reads = 0

        def get_dof_positions(self): return self.q
        def get_dof_velocities(self): return self.q

        def get_root_transforms(self):
            self.root_reads += 1
            return np.array([_xf([1, 0, 0], [1, 0, 0, 0]), _xf([0, 2, 0], [1, 0, 0, 0])], np.float32)

        def get_root_velocities(self):
            self.root_reads += 1
            return np.zeros((2, 6), np.float32)

    view = _View()
    sim_view = types.SimpleNamespace(is_valid=True, create_articulation_view=lambda paths: view)
    batch = ArticulationBatch(["/W/r", "/W/s"], sim_view_factory=lambda: sim_view, logger=logging.getLogger("t"))
    assert batch.sample(1)
    robots = []
    for path in ("/W/r", "/W/s"):
        a, pv = _robot(link_view=None)
        a._path, a._name = path, path.rsplit("/", 1)[-1]
        pv.q[:] = 0.0
        a.set_batch(batch)
        a.synchronize()                                                   # verifies its batch row
        assert a._batch_verified
        robots.append(a)
    msgs = [_root_frames(a.get_actual())[0] for a in robots]
    assert [m["pos"] for m in msgs] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    assert view.root_reads == 2                                           # transforms + velocities, for both robots


def test_batch_root_reads_are_lazy_and_fail_soft():
    class _View:
        prim_paths = ["/W/r"]
        count, max_dofs = 1, 1
        def get_dof_positions(self): return np.zeros((1, 1), np.float32)
        def get_dof_velocities(self): return np.zeros((1, 1), np.float32)
        def get_root_transforms(self): raise RuntimeError("gone")
        def get_root_velocities(self): return np.ones((1, 6), np.float32)

    sim_view = types.SimpleNamespace(is_valid=True, create_articulation_view=lambda paths: _View())
    batch = ArticulationBatch(["/W/r"], sim_view_factory=lambda: sim_view, logger=logging.getLogger("t"))
    assert batch.root_transform(0) is None                                # not sampled yet: no read at all
    assert batch.sample(1)
    assert batch.root_transform(0) is None and batch.root_velocity(5) is None
    assert batch.root_transform(0) is None                                # the failed read is cached for the step
    assert np.allclose(batch.root_velocity(0), 1.0)


# ------------------------------------------------------------------------------------------ root link resolution
def test_root_link_path_falls_back_to_the_first_rigid_body_then_the_prim(monkeypatch):
    a, pv = _robot(link_view=None)
    assert a._root_link_path(a._art) == (ROOT, "root link")
    pv.link_paths = None
    monkeypatch.setattr(articulation.WrappedArticulation, "_first_rigid_body_path",
                        staticmethod(lambda stage, path: path + "/links/body"))
    assert a._root_link_path(a._art, stage=object()) == ("/W/r/links/body", "first rigid body")
    monkeypatch.setattr(articulation.WrappedArticulation, "_first_rigid_body_path",
                        staticmethod(lambda stage, path: None))
    assert a._root_link_path(a._art, stage=object()) == ("/W/r", "articulation prim")


def test_physics_reinit_resolves_the_root_link_again():
    store = _world_store()
    a, pv = _robot(link_view=sampled_link_view(store))
    a.synchronize()
    a.get_actual()
    assert a._root_link is not None
    a._on_physics_reinit(a._art, "test")
    assert a._root_link is None and a._root_row_gen == -1


# ------------------------------------------------------------------------------------------ on the wire
class _PubSocket:
    def __init__(self):
        self.sent = []

    def send(self, frame, flags=0):
        self.sent.append((frame, flags))


def _handler():
    h = interface.MessageHandler(host="127.0.0.1", subscribe_port=1, publish_port=2)
    h._is_server_running = True
    h._publish_socket = _PubSocket()
    return h


def test_moderator_publishes_root_pose_nonblocking_for_articulations_only():
    store = _world_store()
    holder = {"view": FakeSimView(store)}
    lv = LinkStateView(list(store), sim_view_factory=lambda: holder["view"], logger=logging.getLogger("t"))
    robot, _ = _robot(link_view=lv)
    robot._name = "go1"
    device_map = {"Articulation": {"go1": {"path": robot._path}}, "RigidObject": {"ball": {"path": "/W/ball"}}}
    m = make_moderator({"go1": robot, "ball": make_tracked_object(link_view=lv)}, device_map=device_map,
                       link_view=lv, handler=_handler())
    m.synchronize(1 / 250)
    sent = m._message_handler._publish_socket.sent
    assert all(flags == interface.zmq.NOBLOCK for _, flags in sent)
    topics = [frame.split(b"||", 1)[0] for frame, _ in sent]
    assert topics == [b"TM/go1/sim_state", b"TM/go1/root_pose", b"TM/ball/object_state"]
    frame = next(f for f, _ in sent if f.startswith(b"TM/go1/root_pose||"))
    msg = root_pose.decode_root_pose(frame.split(b"||", 1)[1])            # what an observer on :53334 decodes
    assert msg["pos"] == pytest.approx([1.25, -0.5, 0.31], abs=1e-5)
    m.synchronize(1 / 250)                                                # same sim time: not due again
    assert sum(1 for f, _ in sent if b"/root_pose||" in f) == 1


def test_a_full_publish_queue_drops_the_frame_instead_of_blocking():
    class _Full(_PubSocket):
        def send(self, frame, flags=0):
            assert flags == interface.zmq.NOBLOCK
            raise interface.zmq.Again()

    store = _world_store()
    lv = sampled_link_view(store)
    robot, _ = _robot(link_view=lv)
    m = make_moderator({"r": robot}, link_view=lv, handler=_handler())
    m._message_handler._publish_socket = _Full()
    m.synchronize(1 / 250)                                                # no exception, nothing queued
    assert robot._root_seq == 1
