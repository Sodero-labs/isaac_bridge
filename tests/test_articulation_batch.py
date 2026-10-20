"""The stage-wide articulation batch: one tensor view for every robot's joint state and drive targets, so the
per-step backend calls do not grow with the robot count."""
from __future__ import annotations

import logging

import numpy as np

from fakes import command, make_robot
from scripts import protocol
from scripts.articulation_batch import ArticulationBatch
from scripts.device import articulation


class _FakeBatchView:
    """omni.physics.tensors ArticulationView over several articulations, padded to max_dofs."""

    def __init__(self, robots):                     # robots: [(path, [dof names]), ...]
        self.prim_paths = [p for p, _ in robots]
        self.count = len(robots)
        self.max_dofs = max((len(n) for _, n in robots), default=0)
        self.q = np.zeros((self.count, self.max_dofs), dtype=np.float32)
        self.dq = np.zeros((self.count, self.max_dofs), dtype=np.float32)
        self.writes = []                            # (tensor, array copy)
        self.rows = []                              # (tensor, row indices written)
        self.reads = 0
        self.proj = np.zeros((self.count, self.max_dofs), dtype=np.float32)
        self.grav = np.zeros((self.count, self.max_dofs), dtype=np.float32)
        self.cori = np.zeros((self.count, self.max_dofs), dtype=np.float32)
        self.mass = np.zeros((self.count, self.max_dofs, self.max_dofs), dtype=np.float32)
        self.lazy_reads = []

    @property
    def dof_paths(self):                            # reading this on a multi-articulation view crashed Isaac Sim 5.0
        raise AssertionError("ArticulationBatch must never touch ArticulationView.dof_paths")

    def get_dof_positions(self):
        self.reads += 1
        return self.q

    def get_dof_velocities(self):
        self.reads += 1
        return self.dq

    def _write(self, name, data, indices):
        assert data.shape == (self.count, self.max_dofs), data.shape
        assert indices.dtype == np.uint32 and 0 < len(indices) <= self.count and int(indices.max()) < self.count
        self.writes.append((name, np.array(data)))
        self.rows.append((name, [int(i) for i in indices]))    # which articulations that call applied to

    def set_dof_position_targets(self, d, i):
        self._write("pos", d, i)

    def set_dof_velocity_targets(self, d, i):
        self._write("vel", d, i)

    def set_dof_actuation_forces(self, d, i):
        self._write("eff", d, i)

    def set_dof_stiffnesses(self, d, i):
        self._write("kp", d, i)

    def set_dof_dampings(self, d, i):
        self._write("kd", d, i)

    # the reads a step may or may not need
    def get_dof_projected_joint_forces(self):
        self.lazy_reads.append("proj")
        return self.proj

    def get_gravity_compensation_forces(self):
        self.lazy_reads.append("grav")
        return self.grav

    def get_coriolis_and_centrifugal_compensation_forces(self):
        self.lazy_reads.append("cori")
        return self.cori

    def get_generalized_mass_matrices(self):
        self.lazy_reads.append("mass")
        return self.mass


class _FakeSimView:
    def __init__(self, robots):
        self.robots = robots
        self.is_valid = True
        self.created = 0
        self.last = None

    def create_articulation_view(self, paths):
        self.created += 1
        self.last = _FakeBatchView([(p, n) for p, n in self.robots if p in paths])
        return self.last


ROBOTS = [("/World/go1", [f"g{i}" for i in range(12)]),
          ("/World/duck", [f"d{i}" for i in range(14)]),
          ("/World/ur3", [f"u{i}" for i in range(18)])]


def _batch(robots=ROBOTS, holder=None, sampled=False):
    holder = holder if holder is not None else {"view": _FakeSimView(robots)}
    b = ArticulationBatch([p for p, _ in robots], sim_view_factory=lambda: holder["view"],
                          logger=logging.getLogger("t"))
    if sampled:
        assert b.sample(1)
    return b, holder


def _go1():
    return make_robot([f"g{i}" for i in range(12)], path="/World/go1")


# ------------------------------------------------------------------ bind / sample / flush
def test_bind_maps_rows_and_pads_to_max_dofs():
    b, holder = _batch()
    assert b.sample(1) is True and b.bound and b.generation == 1
    assert b.max_dofs == 18 and b.count == 3
    assert b.row("/World/duck") == 1 and b.row("/World/nope") == -1


def test_matches_compares_the_row_against_the_robots_own_reading():
    b, holder = _batch(sampled=True)
    holder["view"].last.q[0, :12] = np.linspace(0, 1, 12)
    b.sample(2)
    assert b.matches(0, np.linspace(0, 1, 12)) is True
    assert b.matches(0, np.linspace(0, 1, 12)[::-1]) is False       # a different DOF order is caught
    assert b.matches(1, np.zeros(14)) is True                       # everything zero: the order cannot differ
    assert b.matches(9, np.zeros(12)) is False                      # row out of range
    assert b.matches(0, []) is False


def test_sample_reads_twice_per_step_regardless_of_robot_count():
    b, holder = _batch()
    assert b.sample(1) and holder["view"].last.reads == 3          # one probe at bind + q and dq
    assert b.sample(1) and holder["view"].last.reads == 3          # same step: no second read
    assert b.sample(2) and holder["view"].last.reads == 5          # + 2 for the stage, whatever the robot count


def test_positions_and_velocities_are_per_robot_slices():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.q[1, :14] = np.arange(14)
    view.dq[2, :18] = -np.arange(18)
    b.sample(2)
    assert np.allclose(b.positions(1, 14), np.arange(14))
    assert np.allclose(b.velocities(2, 18), -np.arange(18))
    assert b.positions(0, 12).shape == (12,)


def test_flush_writes_one_call_per_changed_tensor_for_the_whole_stage():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    b.stage_targets(0, np.ones(12), np.zeros(12))
    b.stage_targets(1, np.full(14, 2.0), np.zeros(14))
    b.stage_gains(2, np.full(18, 35.0), np.full(18, 1.5))
    assert b.flush() == 4                                          # pos, vel, kp, kd — not one per robot
    assert sorted(w[0] for w in view.writes) == ["kd", "kp", "pos", "vel"]
    pos = dict(view.writes)["pos"]
    assert np.allclose(pos[0, :12], 1.0) and np.allclose(pos[1, :14], 2.0) and np.allclose(pos[2], 0.0)
    assert b.flush() == 0 and len(view.writes) == 4                # nothing staged since: no calls


def test_flush_writes_only_the_rows_that_staged_something():
    # The buffers start at zero and a robot no gateway commands never stages anything: writing every row would push
    # kp = kd = 0 and zero targets into it, wiping its USD drive gains.
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    b.stage_gains(1, np.full(14, 35.0), np.full(14, 1.5))
    b.stage_targets(1, np.full(14, 0.3), np.zeros(14))
    assert b.flush() == 4
    assert view.rows == [("pos", [1]), ("vel", [1]), ("kp", [1]), ("kd", [1])]
    b.stage_targets(0, np.ones(12), np.zeros(12))
    b.stage_targets(2, np.ones(18), np.zeros(18))
    assert b.flush() == 2
    assert view.rows[-2:] == [("pos", [0, 2]), ("vel", [0, 2])]
    b.clear_efforts(2)                                             # never staged an effort: nothing to write
    assert b.flush() == 0
    b.stage_targets(2, np.ones(18), np.zeros(18), np.full(18, 2.0))
    assert b.flush() == 3 and view.rows[-1] == ("eff", [2])
    b.clear_efforts(2)
    assert b.flush() == 1 and view.rows[-1] == ("eff", [2])
    # every robot staged -> the full index array
    for r, n in enumerate((12, 14, 18)):
        b.stage_targets(r, np.zeros(n), np.zeros(n))
    assert b.flush() == 2 and view.rows[-1] == ("vel", [0, 1, 2])


def test_flush_includes_efforts_only_when_staged_and_clears_them_once():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    b.stage_targets(0, np.zeros(12), np.zeros(12), np.full(12, 3.0))
    assert b.flush() == 3 and dict(view.writes)["eff"][0, 0] == 3.0
    b.stage_targets(0, np.zeros(12), np.zeros(12))                 # no effort this step
    assert b.flush() == 2                                          # eff not rewritten ...
    b.clear_efforts(0)                                             # ... until the robot stops asking for one
    assert b.flush() == 1
    assert np.allclose(dict(view.writes)["eff"], 0.0)
    b.clear_efforts(0)
    assert b.flush() == 0                                          # already zero: nothing to write


def test_rebind_on_a_new_simulation_view_and_survives_no_physics():
    b, holder = _batch()
    assert b.sample(1) and b.generation == 1
    holder["view"].is_valid = False
    assert b.sample(2) and b.generation == 2
    holder["view"] = _FakeSimView(ROBOTS)
    assert b.sample(3) and b.generation == 3
    holder["view"] = None
    assert b.sample(4) is False and b.ok is False


def test_bind_refuses_a_non_numpy_view():
    class _Weird(_FakeSimView):
        def create_articulation_view(self, paths):
            v = super().create_articulation_view(paths)
            v.get_dof_positions = lambda: object()
            return v

    b, _ = _batch(holder={"view": _Weird(ROBOTS)})
    assert b.sample(1) is False and b.bound is False


def test_flush_failure_drops_the_batch_for_a_rebind():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.set_dof_position_targets = lambda d, i: (_ for _ in ()).throw(RuntimeError("gone"))
    b.stage_targets(0, np.zeros(12), np.zeros(12))
    assert b.flush() == 0 and b.ok is False
    assert b.sample(2) is True and b.generation == 2                # rebound on the next step


# ------------------------------------------------------------------ on-demand reads
def test_measured_efforts_are_one_read_shared_by_every_robot():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.proj[1, :14] = np.arange(14)
    b.sample(2)
    assert np.allclose(b.projected_joint_forces(1, 14), np.arange(14))
    assert np.allclose(b.projected_joint_forces(0, 12), 0.0)      # second robot, same buffer
    assert view.lazy_reads.count("proj") == 1                      # one backend call for the stage
    b.sample(3)
    b.projected_joint_forces(2, 18)
    assert view.lazy_reads.count("proj") == 2                      # a new step reads again


def test_a_step_that_needs_nothing_makes_no_extra_call():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    for step in range(2, 6):
        b.sample(step)
        b.stage_targets(0, np.zeros(12), np.zeros(12))
        b.flush()
    assert view.lazy_reads == []                                   # efforts / gravity never asked for


def test_servo_model_reads_come_from_the_batch():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.grav[0, :12] = 1.5
    view.cori[0, :12] = -0.5
    view.mass[0, :12, :12] = np.eye(12) * 3.0
    b.sample(2)
    assert np.allclose(b.gravity_forces(0, 12), 1.5)
    assert np.allclose(b.coriolis_forces(0, 12), -0.5)
    m = b.mass_matrix(0, 12)
    assert m.shape == (12, 12) and np.allclose(np.diagonal(m), 3.0)
    assert view.lazy_reads == ["grav", "cori", "mass"]


def test_lazy_read_failure_falls_back_to_none_without_raising(caplog):
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.get_dof_projected_joint_forces = lambda: (_ for _ in ()).throw(RuntimeError("gone"))
    b.sample(2)
    with caplog.at_level(logging.WARNING):
        assert b.projected_joint_forces(0, 12) is None
    assert b.projected_joint_forces(0, 12) is None                 # cached failure: no retry inside the step
    assert any("per-robot fallback" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ a robot through the batch
def test_robot_reads_and_writes_through_the_batch_without_touching_its_own_view():
    b, holder = _batch(sampled=True)
    holder["view"].last.q[0, :12] = 0.25
    b.sample(2)
    a, pv = _go1()
    pv.q[0, :] = 0.25                                                # its own view agrees (verification passes)
    a.set_batch(b)
    a.update_target("sim_command", command(12, np.linspace(0, 1, 12)))
    a.synchronize()                                                  # probes its own view once, verifies once
    pv.calls.clear()
    for step in range(3, 6):
        b.sample(step)
        a.update_target("sim_command", command(12, np.linspace(0, 1, 12)))
        a.synchronize()
    assert pv.calls == []                                            # no per-robot backend call at all
    assert np.allclose(a._actual[0], 0.25)                           # the joint state came from the batch row
    assert b.flush() == 4
    pos = dict(holder["view"].last.writes)["pos"]
    assert np.allclose(pos[0, :12], np.linspace(0, 1, 12)) and np.allclose(pos[0, 12:], 0.0)


def test_robot_falls_back_when_its_row_does_not_report_its_joint_state(caplog):
    b, holder = _batch(sampled=True)
    a, pv = _go1()
    a.set_batch(b)
    pv.q[0, :] = np.linspace(0, 1, 12)                               # the robot's own view says something else
    b.sample(2)                                                      # the batch row stays at zero
    with caplog.at_level(logging.WARNING):
        a.update_target("sim_command", command(12, np.zeros(12)))
        a.synchronize()
    assert a._batch_row == -1
    assert "pos" in pv.writes()                                      # drove through its own view instead
    assert any("does not report this robot" in r.getMessage() for r in caplog.records)


def test_robot_reverifies_its_row_periodically(monkeypatch):
    monkeypatch.setattr(articulation, "_BATCH_RECHECK_STEPS", 3)
    b, holder = _batch(sampled=True)
    a, pv = _go1()
    a.set_batch(b)
    a.synchronize()                                                  # verifies (one read through its own view)
    assert a._batch_verified and a._batch_row == 0
    reads = pv.calls.count("get_q")
    for step in range(2, 5):
        b.sample(step)
        a.synchronize()
    assert pv.calls.count("get_q") == reads                          # no per-robot read while the check holds
    b.sample(5)
    a.synchronize()
    assert pv.calls.count("get_q") == reads + 1                      # ... then one again


def test_robot_without_a_batch_uses_its_own_view():
    a, pv = _go1()
    a.set_batch(None)
    a.update_target("sim_command", command(12, np.zeros(12)))
    a.synchronize()                                                  # the first step also probes the view once
    pv.calls.clear()
    a.update_target("sim_command", command(12, np.ones(12)))
    a.synchronize()
    assert [c if isinstance(c, str) else c[0] for c in pv.calls] == ["get_q", "get_dq", "pos", "vel"]   # gains unchanged


def test_batch_and_per_robot_paths_produce_the_same_targets():
    q_d = np.linspace(-0.5, 0.5, 12)
    a1, pv1 = _go1()
    a1.set_batch(None)
    a1.update_target("sim_command", command(12, q_d))
    a1.synchronize()
    sent_direct = pv1.writes()

    b, holder = _batch(sampled=True)
    a2, _ = _go1()
    a2.set_batch(b)
    a2.update_target("sim_command", command(12, q_d))
    a2.synchronize()
    b.flush()
    staged = dict(holder["view"].last.writes)
    assert sorted(sent_direct) == sorted(staged)                     # the same tensors written
    assert np.allclose(staged["pos"][0, :12], q_d, atol=1e-6)
    assert np.allclose(a1._actual[0], a2._actual[0])


def test_effort_is_cleared_once_when_a_robot_stops_asking_for_one():
    b, holder = _batch(sampled=True)
    a, _ = _go1()
    a.set_batch(b)
    a.update_target("sim_command", command(12, mode=protocol.MODE_TORQUE, tau_d=np.full(12, 0.4)))
    a.synchronize()
    b.flush()
    assert "eff" in dict(holder["view"].last.writes)
    holder["view"].last.writes.clear()
    a.update_target("sim_command", command(12, np.zeros(12)))         # back to POSITION: no feed-forward
    a.synchronize()
    b.flush()
    assert np.allclose(dict(holder["view"].last.writes)["eff"], 0.0)


def test_physics_reinit_rechecks_the_batch_row():
    b, holder = _batch(sampled=True)
    a, _ = _go1()
    a.set_batch(b)
    a.synchronize()
    assert a._batch_row == 0
    a._on_physics_reinit(a._art, "test")
    assert a._batch_gen == -1                                        # re-resolved at the next step


def test_robot_reads_measured_efforts_through_the_batch():
    b, holder = _batch(sampled=True)
    view = holder["view"].last
    view.proj[0, :12] = np.linspace(1, 2, 12)
    b.sample(2)
    a, _ = _go1()
    a.set_batch(b)
    assert a._batch_view(a._art) is b                                # resolves and verifies the row
    calls = []
    a._art.get_measured_joint_efforts = lambda: calls.append("wrapper") or np.zeros((1, 12), dtype=np.float32)
    tau = a._read_measured_efforts(a._art, b)
    assert np.allclose(tau, np.linspace(1, 2, 12)) and calls == []
    assert np.allclose(a._read_measured_efforts(a._art, None), 0.0) and calls == ["wrapper"]
