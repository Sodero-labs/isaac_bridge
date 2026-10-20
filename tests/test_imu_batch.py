"""The stage-wide IMU pass: one vectorized computation for every robot, equal to the per-robot path."""
from __future__ import annotations

import logging
import types

import numpy as np

from fakes import FakeSimView, make_robot, rand_quat
from scripts import protocol
from scripts.imu_batch import ImuBatch, quat_to_matrix_wxyz
from scripts.link_view import LinkStateView

ROBOTS = ("go1", "duck", "humanoid")


def _world(rng):
    """path -> (transform xyz xyzw, velocity) for each robot's IMU link and two contact links."""
    store = {}
    for r in ROBOTS:
        for suffix in ("base", "c0", "c1"):
            q = rand_quat(rng)
            store[f"/World/{r}/links/{suffix}"] = (
                np.concatenate([rng.standard_normal(3), q[[1, 2, 3, 0]]]).astype(np.float32),
                rng.standard_normal(6).astype(np.float32))
    return store


def _advance(store, rng, scale=0.02):
    for path, (xf, vel) in store.items():
        xf = xf.copy()
        xf[:3] += rng.standard_normal(3).astype(np.float32) * scale
        store[path] = (xf, (vel + rng.standard_normal(6).astype(np.float32) * scale).astype(np.float32))


def _spec(robot, *, tau=0.02, height=0.01, ground=0.0):
    return {
        "path": f"/World/{robot}/links/base",
        "contact_paths": [f"/World/{robot}/links/c0", f"/World/{robot}/links/c1"],
        "offset": np.array([0.01, -0.02, 0.05]),
        "com_local": np.array([0.0144, -0.0017, 0.1495]),
        "contact_offsets": [np.array([0.0, 0.0, -0.21]), np.array([0.0, 0.0, -0.21])],
        "contact_height": height, "ground_z": ground, "accel_tau": tau,
    }


def _imu_robot(robot, link_view, spec):
    """A robot whose IMU is computed per robot (the reference), through `link_view`."""
    a, _ = make_robot(path=f"/World/{robot}", sim_time=0.0)
    a._link_view = link_view
    a._imu = {
        "prim": None, "offset": spec["offset"], "path": spec["path"],
        "com_local": spec["com_local"],
        "contacts": [(None, off) for off in spec["contact_offsets"]],
        "contact_paths": spec["contact_paths"],
        "contact_height": spec["contact_height"], "ground_z": spec["ground_z"],
        "prev_v": None, "prev_t": None, "accel_tau": spec["accel_tau"],
    }
    return a


def _link_view(store, step=0):
    sim_view = FakeSimView(store)              # one simulation view for the whole run, as during a play
    view = LinkStateView(list(store), sim_view_factory=lambda: sim_view, logger=logging.getLogger("t"))
    assert view.sample(step)
    return view


def test_quaternion_to_matrix():
    rng = np.random.default_rng(5)
    q = np.array([rand_quat(rng) for _ in range(20)])
    R = quat_to_matrix_wxyz(q)
    assert R.shape == (20, 3, 3)
    assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)     # orthonormal
    assert np.allclose(np.linalg.det(R), 1.0)
    h = np.sqrt(0.5)
    np.testing.assert_allclose(quat_to_matrix_wxyz(np.array([h, 0.0, 0.0, h])) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                               atol=1e-12)                                          # +90 deg yaw turns x into y


def test_rotating_link_keeps_an_imu_at_its_origin_stationary():
    # pure rotation about the link origin: the COM moves, an IMU at the origin does not; a 90 degree yaw makes the
    # COM correction go through the rotation
    h = np.sqrt(.5)
    xf = np.array([[2., 3., 1., 0., 0., h, h]], dtype=np.float32)
    vel = np.array([[-.4, 0., 0., 0., 0., 2.]], dtype=np.float32)
    view = types.SimpleNamespace(ok=True, generation=1, row=lambda p: 0, transforms=lambda: xf, velocities=lambda: vel)
    spec = dict(path="/base", offset=np.zeros(3), com_local=np.array([.2, 0., 0.]))
    dev = _imu_robot("test", None, _spec("test"))
    batch = ImuBatch([(dev, spec)])
    assert batch.compute(view, 0.)
    state = protocol.parse_imu_state(dev._imu_payload)
    np.testing.assert_allclose(state.pos_w, [2., 3., 1.], atol=1e-7)
    np.testing.assert_allclose(state.lin_vel_w, [0., 0., 0.], atol=1e-7)
    np.testing.assert_allclose(state.gyro, [0., 0., 2.], atol=1e-7)


def test_batch_matches_the_per_robot_pass_over_many_steps():
    rng = np.random.default_rng(11)
    store = _world(rng)
    view = _link_view(store)
    specs = {r: _spec(r, tau=0.01 * (i + 1)) for i, r in enumerate(ROBOTS)}
    ref = {r: _imu_robot(r, view, specs[r]) for r in ROBOTS}          # per-robot reference
    bat = {r: _imu_robot(r, view, specs[r]) for r in ROBOTS}          # driven by the batch
    batch = ImuBatch([(bat[r], specs[r]) for r in ROBOTS], logger=logging.getLogger("t"))
    t = 0.0
    for step in range(1, 8):
        _advance(store, rng)
        view.sample(step)
        t += 0.004
        for r in ROBOTS:
            ref[r]._sim_context.current_time = t
            ref[r]._sample_imu()
        assert batch.compute(view, t) is True
        for r in ROBOTS:
            assert len(bat[r]._imu_payload) == protocol.IMU_STATE_SIZE and bat[r]._imu_payload[-6:] == bytes(6)
            got, want = protocol.parse_imu_state(bat[r]._imu_payload), protocol.parse_imu_state(ref[r]._imu_payload)
            assert got.seq == want.seq == step
            assert got.sim_time_ns == want.sim_time_ns
            for field in ("quat_xyzw", "gyro", "accel", "lin_vel_w", "pos_w"):
                assert np.allclose(getattr(got, field), getattr(want, field), atol=1e-9), (r, step, field)
            assert np.array_equal(got.contacts, want.contacts), (r, step)


def test_contacts_follow_each_robots_own_height_and_ground():
    rng = np.random.default_rng(12)
    store = _world(rng)
    for r in ROBOTS:                                   # feet flat, identity rotation, known heights
        for j, suffix in enumerate(("c0", "c1")):
            xf = np.array([0, 0, 0.2 + 0.1 * j, 0, 0, 0, 1], dtype=np.float32)
            store[f"/World/{r}/links/{suffix}"] = (xf, np.zeros(6, dtype=np.float32))
    view = _link_view(store, step=1)
    specs = {"go1": _spec("go1", height=0.02), "duck": _spec("duck", height=0.2),
             "humanoid": _spec("humanoid", height=0.02, ground=0.1)}
    devs = {r: _imu_robot(r, view, specs[r]) for r in ROBOTS}
    batch = ImuBatch([(devs[r], specs[r]) for r in ROBOTS], logger=logging.getLogger("t"))
    assert batch.compute(view, 0.004)
    # sole z = link z - 0.21 -> c0 at -0.01, c1 at +0.09
    assert list(protocol.parse_imu_state(devs["go1"]._imu_payload).contacts) == [1, 0]       # threshold 0.02
    assert list(protocol.parse_imu_state(devs["duck"]._imu_payload).contacts) == [1, 1]      # threshold 0.2
    assert list(protocol.parse_imu_state(devs["humanoid"]._imu_payload).contacts) == [1, 1]  # ground at 0.1


def test_missing_link_disables_the_batch_without_raising(caplog):
    rng = np.random.default_rng(13)
    store = _world(rng)
    del store["/World/duck/links/c1"]                  # one contact link never made it into the view
    view = _link_view(store, step=1)
    specs = {r: _spec(r) for r in ROBOTS}
    devs = {r: _imu_robot(r, view, specs[r]) for r in ROBOTS}
    batch = ImuBatch([(devs[r], specs[r]) for r in ROBOTS], logger=logging.getLogger("t"))
    with caplog.at_level(logging.INFO):
        assert batch.compute(view, 0.004) is False
    assert all(d._imu_payload is None for d in devs.values())
    assert batch.compute(view, 0.008) is False        # stays off for this view generation, no repeated work


def test_empty_batch_and_unbound_view_are_safe():
    assert ImuBatch([], logger=logging.getLogger("t")).compute(None, 0.0) is False
    view = LinkStateView(["/World/go1/links/base"], sim_view_factory=lambda: None, logger=logging.getLogger("t"))
    assert view.sample(1) is False
    assert ImuBatch([], logger=logging.getLogger("t")).compute(view, 0.0) is False


def test_filter_state_is_per_robot():
    rng = np.random.default_rng(14)
    store = _world(rng)
    view = _link_view(store)
    specs = {"go1": _spec("go1", tau=0.0), "duck": _spec("duck", tau=1.0), "humanoid": _spec("humanoid", tau=0.02)}
    devs = {r: _imu_robot(r, view, specs[r]) for r in ROBOTS}
    batch = ImuBatch([(devs[r], specs[r]) for r in ROBOTS], logger=logging.getLogger("t"))
    accels = {r: [] for r in ROBOTS}
    t = 0.0
    for step in range(1, 6):
        _advance(store, rng, scale=0.05)
        view.sample(step)
        t += 0.004
        batch.compute(view, t)
        for r in ROBOTS:
            accels[r].append(protocol.parse_imu_state(devs[r]._imu_payload).accel.copy())
    spread = {r: float(np.std([np.linalg.norm(a) for a in v])) for r, v in accels.items()}
    assert spread["go1"] > spread["duck"]              # tau 0 follows every sample, tau 1.0 barely moves
