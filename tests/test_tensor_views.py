"""The stage-wide link view and a robot's raw tensor view, against fakes of the omni.physics.tensors views."""
from __future__ import annotations

import logging

import numpy as np

from fakes import FakeSimView, command, make_robot, make_tracked_object, rand_quat, random_store
from scripts import protocol
from scripts.link_view import LinkStateView


# ------------------------------------------------------------------ link view
def test_link_view_binds_lazily_and_samples_once_per_step():
    rng = np.random.default_rng(1)
    paths = ["/W/r/links/a", "/W/r/links/b", "/W/ball"]
    store = random_store(paths, rng)
    holder = {"view": None}
    lv = LinkStateView(paths, sim_view_factory=lambda: holder["view"], logger=logging.getLogger("t"))
    assert lv.sample(1) is False and not lv.bound and lv.transforms() is None
    holder["view"] = FakeSimView(store, order=["/W/ball", "/W/r/links/b", "/W/r/links/a"])   # backend order differs
    assert lv.sample(2) is True and lv.bound and lv.generation == 1
    xf = lv.transforms()
    assert xf.shape == (3, 7) and lv.paths == paths                       # our order, whatever the backend did
    assert np.allclose(xf[2], store["/W/ball"][0])
    pos, quat = lv.pose_wxyz(lv.row("/W/r/links/b"))
    assert np.allclose(pos, store["/W/r/links/b"][0][:3])
    assert np.allclose(quat, store["/W/r/links/b"][0][[6, 3, 4, 5]], atol=1e-6)   # xyzw -> wxyz
    assert np.allclose(lv.velocity(lv.row("/W/ball")), store["/W/ball"][1], atol=1e-6)
    reads = lv._view.reads
    assert lv.sample(2) is True and lv._view.reads == reads               # same step: no second backend read
    assert lv.sample(3) is True and lv._view.reads == reads + 1
    assert lv.row("/nope") == -1


def test_link_view_rebinds_when_the_simulation_view_is_replaced_or_invalid():
    rng = np.random.default_rng(2)
    paths = ["/W/a", "/W/b"]
    holder = {"view": FakeSimView(random_store(paths, rng))}
    lv = LinkStateView(paths, sim_view_factory=lambda: holder["view"], logger=logging.getLogger("t"))
    assert lv.sample(1) and lv.generation == 1
    holder["view"].is_valid = False                                     # timeline stop: PhysX dropped the scene
    assert lv.sample(2) and lv.generation == 2 and holder["view"].created == 2
    holder["view"] = FakeSimView(random_store(paths, rng))              # play: a brand-new SimulationView
    assert lv.sample(3) and lv.generation == 3
    holder["view"] = None                                               # no physics at all
    assert lv.sample(4) is False and lv.ok is False and lv.transforms() is None


def test_link_view_drops_paths_the_backend_does_not_know():
    rng = np.random.default_rng(4)
    store = random_store(["/W/a", "/W/c"], rng)
    lv = LinkStateView(["/W/a", "/W/b", "/W/c"], sim_view_factory=lambda: FakeSimView(store),
                       logger=logging.getLogger("t"))
    assert lv.sample(1)
    assert lv.paths == ["/W/a", "/W/c"] and lv.row("/W/b") == -1 and lv.transforms().shape == (2, 7)


# ------------------------------------------------------------------ a robot's raw tensor view
def test_raw_view_targets_are_written_once_per_command_not_every_step():
    a, pv = make_robot()
    a.synchronize()                                        # no command yet: reads only
    assert pv.writes() == [] and a._actual is not None
    a.update_target("sim_command", command(3, [1.0, 2.0, 3.0]))
    pv.calls.clear()
    a.synchronize()
    assert pv.writes() == ["kp", "kd", "pos", "vel"]                    # gains + targets, no zero efforts
    assert np.allclose(pv.last("pos")[0], [1.0, 2.0, 3.0])
    pv.calls.clear()
    a.synchronize()
    a.synchronize()
    assert pv.writes() == []                                            # unchanged command: PhysX keeps its targets
    assert pv.calls == ["get_q", "get_dq", "get_q", "get_dq"]
    a.update_target("sim_command", command(3, [1.0, 2.5, 3.0]))
    pv.calls.clear()
    a.synchronize()
    assert pv.writes() == ["pos", "vel"]                                # same gains: only the targets


def test_raw_view_efforts_are_reapplied_every_step_while_requested():
    a, pv = make_robot()
    a.update_target("sim_command", command(3, mode=protocol.MODE_TORQUE, tau_d=[0.5, 0.0, -0.5]))
    a.synchronize()
    pv.calls.clear()
    a.synchronize()
    assert pv.writes() == ["pos", "vel", "eff"]
    assert np.allclose(pv.last("eff")[0], [0.5, 0.0, -0.5])


def test_raw_view_falls_back_to_the_wrapper_when_not_numpy():
    a, pv = make_robot()
    pv.get_dof_positions = lambda: object()                 # a torch- or warp-backed view: not usable here
    calls = []
    a._art.get_joint_positions = lambda: calls.append("q") or np.zeros((1, 3))
    a._art.get_joint_velocities = lambda: calls.append("dq") or np.zeros((1, 3))
    a.synchronize()
    assert calls == ["q", "dq"] and a._raw_ok is False


def test_physics_reinit_and_resume_mark_the_command_dirty():
    a, pv = make_robot()
    a.update_target("sim_command", command(3, [1.0, 2.0, 3.0]))
    a.synchronize()
    assert a._cmd_dirty is False
    a.on_resume()
    assert a._cmd_dirty is True
    a._cmd_dirty = False
    a._on_physics_reinit(a._art, "test")
    assert a._cmd_dirty is True and a._raw_ok is None


# ------------------------------------------------------------------ IMU through the link view
class _FakePrim:
    def __init__(self, pose, vel):
        self._pose, self._vel = pose, vel

    def get_world_poses(self):
        p, q = self._pose
        return np.asarray([p]), np.asarray([q])

    def get_velocities(self):
        return np.asarray([self._vel])


def _imu_robot(store, paths):
    a, _ = make_robot()
    imu_path, c0, c1 = paths
    a._imu = {
        "prim": _FakePrim(*store[imu_path]), "offset": np.array([0.01, 0.0, 0.05]),
        "contacts": [(_FakePrim(*store[c0]), np.array([0, 0, -0.2])), (_FakePrim(*store[c1]), np.array([0, 0, -0.2]))],
        "path": imu_path, "contact_paths": [c0, c1], "contact_height": 0.03, "ground_z": 0.0,
        "prev_v": None, "prev_t": None, "accel_tau": 0.02,
    }
    return a


def test_imu_from_the_link_view_matches_the_rigid_prim_path():
    rng = np.random.default_rng(9)
    paths = ["/W/r/links/base", "/W/r/links/lf", "/W/r/links/rf"]
    store = {p: ((rng.standard_normal(3), rand_quat(rng)), rng.standard_normal(6)) for p in paths}
    a_prim = _imu_robot(store, paths)
    a_view = _imu_robot(store, paths)
    view_store = {p: (np.concatenate([store[p][0][0], store[p][0][1][[1, 2, 3, 0]]]), store[p][1]) for p in paths}
    lv = LinkStateView(paths, sim_view_factory=lambda: FakeSimView(view_store), logger=logging.getLogger("t"))
    assert lv.sample(1)
    a_view.set_link_view(lv)
    for step in range(3):
        a_prim._sample_imu()
        a_view._sample_imu()
        assert a_view._imu_payload is not None
        got, ref = protocol.parse_imu_state(a_view._imu_payload), protocol.parse_imu_state(a_prim._imu_payload)
        assert got.seq == ref.seq and got.sim_time_ns == ref.sim_time_ns
        for field in ("quat_xyzw", "gyro", "accel", "lin_vel_w", "pos_w"):
            assert np.allclose(getattr(got, field), getattr(ref, field), atol=2e-5), field   # float32 view vs float64
        assert np.array_equal(got.contacts, ref.contacts)
        for p in paths:                                   # move everything a bit between steps
            pose, vel = store[p]
            store[p] = ((pose[0] + 0.01, pose[1]), vel + 0.1)
            view_store[p] = (np.concatenate([store[p][0][0], store[p][0][1][[1, 2, 3, 0]]]), store[p][1])
            a_prim._imu["prim"] = _FakePrim(*store[paths[0]])
            a_prim._imu["contacts"] = [(_FakePrim(*store[paths[1]]), np.array([0, 0, -0.2])),
                                       (_FakePrim(*store[paths[2]]), np.array([0, 0, -0.2]))]
            a_prim._sim_context.current_time += 0.004
            a_view._sim_context.current_time += 0.004
        lv.sample(step + 2)


# ------------------------------------------------------------------ tracked object through the link view
def test_tracked_object_samples_through_the_view_only_when_a_publish_is_due():
    obj = make_tracked_object()                             # no RigidPrim: the view must carry it
    rng = np.random.default_rng(10)
    store = random_store(["/W/ball"], rng)
    lv = LinkStateView(["/W/ball"], sim_view_factory=lambda: FakeSimView(store), logger=logging.getLogger("t"))
    lv.sample(1)
    obj.set_link_view(lv)
    obj.synchronize()
    frames = obj.get_actual()
    assert len(frames) == 1 and frames[0][0] == "object_state"
    obj._sim_context.current_time = 0.004                   # 4 ms later: not due (50 Hz)
    obj._state = None
    obj.synchronize()
    assert obj._state is None
    obj._sim_context.current_time = 0.021
    obj.synchronize()
    assert obj._state is not None and np.allclose(obj._state[0], store["/W/ball"][0][:3])
