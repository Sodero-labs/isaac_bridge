"""One PhysX tensor view over every articulation of the stage: joint state read once per physics step and drive
targets written once, for all robots together.

Each call into the tensor backend has a fixed cost that does not depend on how many DOFs it moves, so one view
per robot makes the per-step cost grow with the robot count. `create_articulation_view([...])` accepts several
articulations, pads their DOFs to `max_dofs` and exchanges (count, max_dofs) arrays, so the same work is a handful
of backend calls for the whole stage. Each WrappedArticulation keeps its own row and slices or scatters numpy views
of the shared buffers:

    moderator step:  batch.sample(step)        one read for q, one for dq (all robots)
      each robot:    batch.positions(row, n)   a numpy view, no backend call
                     batch.stage_targets(...)  writes into the shared buffer
    moderator end:   batch.flush()             one call per tensor that any robot changed

Reads that a step may or may not need (measured efforts, the servo model's gravity / Coriolis / mass matrix, root
transforms) are lazy: the first robot that asks triggers one backend call for the whole stage and every other robot
reads its row of the same buffer. Only the rare per-robot work (teleport, clean respawn) uses the robot's own view.

Each robot finds its row through the view's `prim_paths` and verifies its DOF order by comparing the joint
positions of that row with what its own view reports, at bind and periodically afterwards. A mismatch drops the
robot back to its own view rather than driving the wrong joints.

Never read `ArticulationView.dof_paths` here: on a multi-articulation view it crashed Isaac Sim 5.0
(omni.physics.tensors 107.3.18) inside the tensor backend. `prim_paths`, `count` and `max_dofs` are safe.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .link_view import _simulation_view

if TYPE_CHECKING:
    from typing import Dict, List, Sequence

_LOG = logging.getLogger("isaac_bridge")


class ArticulationBatch:
    """Shared (count, max_dofs) joint state and drive targets for every articulation of the stage."""

    def __init__(self, paths: "Sequence[str]", *, sim_view_factory=None, logger=None):
        self.paths: "List[str]" = list(dict.fromkeys(paths))
        self._sim_view_factory = sim_view_factory or _simulation_view
        self._logger = logger or _LOG
        self._sim_view = None
        self._view = None
        self.generation = 0
        self.row_of: "Dict[str, int]" = {}
        self.max_dofs = 0
        self.count = 0
        self._q = None              # (count, max_dofs) float64 copy of the last read
        self._dq = None
        self._pos_t = None          # staged targets, kept between steps (PhysX holds them too)
        self._vel_t = None
        self._eff = None
        self._kp = None
        self._kd = None
        self._idx = None            # every row, for a write that every robot staged
        self._dirty = {}            # pos/vel/eff/kp/kd -> the rows staged since the last flush (only those are written)
        self._lazy = {}             # per-step cache of the on-demand reads (see _lazy_read)
        self._sampled_step = None
        self._ok = False
        self._bind_failures = 0

    # -- binding --------------------------------------------------------------
    @property
    def bound(self) -> bool:
        return self._view is not None

    @property
    def ok(self) -> bool:
        return self._ok

    def bind(self) -> bool:
        sim_view = self._sim_view_factory()
        if sim_view is None or not self.paths:
            self._drop()
            return False
        try:
            view = sim_view.create_articulation_view(list(self.paths))
            prim_paths = [str(p) for p in view.prim_paths]
            count = int(view.count)
            max_dofs = int(view.max_dofs)
            probe = view.get_dof_positions()          # never view.dof_paths: see the module docstring
        except Exception as exc:
            self._drop()
            self._bind_failures += 1
            if self._bind_failures in (1, 10, 100):
                self._logger.warning("[BATCH VIEW] bind failed (%s): %s: %s",
                                     self._bind_failures, type(exc).__name__, exc)
            return False
        if not isinstance(probe, np.ndarray) or probe.ndim != 2 or probe.shape[0] != count:
            self._drop()
            self._logger.info("[BATCH VIEW] tensor view is not numpy-backed (%s); per-robot views",
                              type(probe).__name__)
            self._bind_failures += 1
            return False
        self.row_of = {p: i for i, p in enumerate(prim_paths[:count])}
        self.max_dofs = max_dofs
        self.count = count
        self._sim_view = sim_view
        self._view = view
        shape = (count, max_dofs)
        self._q = np.zeros(shape, dtype=np.float64)
        self._dq = np.zeros(shape, dtype=np.float64)
        self._pos_t = np.zeros(shape, dtype=np.float32)
        self._vel_t = np.zeros(shape, dtype=np.float32)
        self._eff = np.zeros(shape, dtype=np.float32)
        self._kp = np.zeros(shape, dtype=np.float32)
        self._kd = np.zeros(shape, dtype=np.float32)
        self._idx = np.arange(count, dtype=np.uint32)
        self._dirty.clear()
        self._lazy.clear()
        self._sampled_step = None
        self._ok = False
        self._bind_failures = 0
        self.generation += 1
        self._logger.info("[BATCH VIEW] bound %d articulations, max_dofs=%d (generation %d): %s",
                          count, max_dofs, self.generation, list(self.row_of))
        return True

    def _drop(self) -> None:
        self._view = None
        self._sim_view = None
        self._ok = False
        self._dirty.clear()

    def _needs_rebind(self) -> bool:
        if self._view is None:
            return True
        sim_view = self._sim_view_factory()
        if sim_view is None or sim_view is not self._sim_view:
            return True
        return not bool(getattr(sim_view, "is_valid", True))

    @property
    def sampled_step(self):
        """The step id of the buffers currently held (devices resolve themselves once per step)."""
        return self._sampled_step

    def row(self, path: str) -> int:
        return self.row_of.get(path, -1)

    def matches(self, row: int, q_ref) -> bool:
        """True when this row's joint positions equal what the robot's own view reports.

        This check stands in for reading `dof_paths` (see the module docstring): with a wrong row or DOF order
        the two readings differ — except while every value is identical (a robot standing at zero), where a wrong
        order is also harmless. Callers repeat it periodically, so a robot that starts moving is checked again
        with values that do tell the orders apart.
        """
        if not self._ok or row < 0 or row >= self.count:
            return False
        q = np.asarray(q_ref, dtype=np.float64).reshape(-1)
        n = min(len(q), self.max_dofs)
        if n == 0:
            return False
        return bool(np.allclose(self._q[row, :n], q[:n], rtol=0.0, atol=1e-5))

    # -- per step ---------------------------------------------------------------
    def sample(self, step_id) -> bool:
        """Read every articulation's joint positions and velocities (2 backend calls for the stage)."""
        if step_id == self._sampled_step:
            return self._ok
        if self._needs_rebind() and not self.bind():
            self._sampled_step = step_id
            self._ok = False
            return False
        self._sampled_step = step_id
        try:
            q = self._view.get_dof_positions()
            dq = self._view.get_dof_velocities()
        except Exception as exc:
            self._logger.warning("[BATCH VIEW] sample failed, rebinding next step: %s: %s",
                                 type(exc).__name__, exc)
            self._drop()
            return False
        np.copyto(self._q, q)          # the view reuses its buffer; the devices keep slices for a whole step
        np.copyto(self._dq, dq)
        self._lazy.clear()             # the on-demand reads belong to the step that asks for them
        self._ok = True
        return True

    # -- on-demand per-step reads (one backend call for the stage, the first asker pays) ------------------
    def _lazy_read(self, key: str, method: str):
        """Read `method` once per sampled step and hand back the whole (count, ...) buffer (None on failure)."""
        if not self._ok:
            return None
        if key in self._lazy:
            return self._lazy[key]
        try:
            data = np.asarray(getattr(self._view, method)())
        except Exception as exc:
            self._logger.warning("[BATCH VIEW] %s failed, per-robot fallback: %s: %s",
                                 method, type(exc).__name__, exc)
            data = None
        self._lazy[key] = data
        return data

    def _row_slice(self, key: str, method: str, row: int, n_dof: int):
        data = self._lazy_read(key, method)
        if data is None or row < 0 or row >= len(data):
            return None
        return data[row][:n_dof]

    def projected_joint_forces(self, row: int, n_dof: int):
        """Measured joint efforts (what Isaac's get_measured_joint_efforts returns), this robot's DOFs."""
        return self._row_slice("proj", "get_dof_projected_joint_forces", row, n_dof)

    def gravity_forces(self, row: int, n_dof: int):
        return self._row_slice("grav", "get_gravity_compensation_forces", row, n_dof)

    def coriolis_forces(self, row: int, n_dof: int):
        return self._row_slice("cori", "get_coriolis_and_centrifugal_compensation_forces", row, n_dof)

    def root_transform(self, row: int):
        """(7,) world transform of this robot's root link, x y z qx qy qz qw, or None."""
        data = self._lazy_read("root", "get_root_transforms")
        if data is None or row < 0 or row >= len(data):
            return None
        return data[row]

    def root_velocity(self, row: int):
        """(6,) world velocity of this robot's root link, linear (COM) then angular, or None."""
        data = self._lazy_read("root_vel", "get_root_velocities")
        if data is None or row < 0 or row >= len(data):
            return None
        return data[row]

    def mass_matrix(self, row: int, n_dof: int):
        """(n_dof, n_dof) generalized mass matrix of this robot, or None."""
        data = self._lazy_read("mass", "get_generalized_mass_matrices")
        if data is None or row < 0 or row >= len(data):
            return None
        m = data[row]
        return m[:n_dof, :n_dof] if m.ndim == 2 else None

    def positions(self, row: int, n_dof: int) -> np.ndarray:
        return self._q[row, :n_dof]

    def velocities(self, row: int, n_dof: int) -> np.ndarray:
        return self._dq[row, :n_dof]

    def _stage(self, key: str, row: int) -> None:
        self._dirty.setdefault(key, set()).add(row)

    def stage_gains(self, row: int, kp_dof: np.ndarray, kd_dof: np.ndarray) -> None:
        n = len(kp_dof)
        self._kp[row, :n] = kp_dof
        self._kd[row, :n] = kd_dof
        self._stage("kp", row)
        self._stage("kd", row)

    def stage_targets(self, row: int, pos_dof: np.ndarray, vel_dof: np.ndarray, eff_dof=None) -> None:
        n = len(pos_dof)
        self._pos_t[row, :n] = pos_dof
        self._vel_t[row, :n] = vel_dof
        self._stage("pos", row)
        self._stage("vel", row)
        if eff_dof is not None:
            self._eff[row, :len(eff_dof)] = eff_dof
            self._stage("eff", row)

    def clear_efforts(self, row: int) -> None:
        """A robot that stops asking for feed-forward effort must not keep the last one applied."""
        if self._eff is not None and np.any(self._eff[row]):
            self._eff[row] = 0.0
            self._stage("eff", row)

    def flush(self) -> int:
        """Push whatever the devices staged this step: one backend call per changed tensor, applied to the rows
        that staged it (the tensor API takes the whole (count, max_dofs) buffer plus the row indices to write).

        Never every row: the buffers start at zero and a robot no gateway commands never stages anything, so a
        write of every row would push kp = kd = 0 and zero targets into it, wiping the drive gains its USD
        authored."""
        if not self._ok or not self._dirty:
            return 0
        writes = (("pos", self._pos_t, "set_dof_position_targets"),
                  ("vel", self._vel_t, "set_dof_velocity_targets"),
                  ("eff", self._eff, "set_dof_actuation_forces"),
                  ("kp", self._kp, "set_dof_stiffnesses"),
                  ("kd", self._kd, "set_dof_dampings"))
        n = 0
        for key, buf, method in writes:
            rows = self._dirty.get(key)
            if not rows:
                continue
            idx = self._idx if len(rows) == self.count else np.fromiter(sorted(rows), dtype=np.uint32, count=len(rows))
            try:
                getattr(self._view, method)(buf, idx)
                n += 1
            except Exception as exc:
                self._logger.warning("[BATCH VIEW] %s failed: %s: %s", method, type(exc).__name__, exc)
                self._drop()
                return n
        self._dirty.clear()
        return n
