"""One PhysX tensor view over every tracked rigid body (robot links and free bodies), sampled once per step.

Every per-step consumer of a link pose — the IMU emulation and its foot contacts, root_pose, tracked objects —
reads its row of this view instead of making its own round trip through Isaac's prim wrappers, each of which
costs index bookkeeping, copies and quaternion shuffles. The whole stage is two backend calls per physics step:
`RigidBodyView.get_transforms()` and `get_velocities()` over all tracked bodies, read straight from the numpy
buffers the tensor frontend owns.

The view is (re)bound lazily: PhysX rebuilds its scene on every timeline play, which invalidates every tensor
view (the SimulationManager hands out a new SimulationView); `sample()` notices the new view object (or a failed
read) and binds again. Nothing here touches USD.

Contract for consumers: `transforms()` / `velocities()` return the frontend's reused buffers, valid until the
next `sample()`; copy what must survive a step.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Sequence, Tuple

_LOG = logging.getLogger("isaac_bridge")


def _simulation_view():
    """The live omni.physics.tensors SimulationView (None before the first physics warm-up)."""
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except Exception:  # pragma: no cover - Isaac Sim not importable
        return None
    try:
        return SimulationManager.get_physics_sim_view()
    except Exception:
        return None


class LinkStateView:
    """World transforms + velocities of a fixed set of rigid-body prims, one backend call each per step."""

    def __init__(self, paths: "Sequence[str]", *, sim_view_factory=None, logger=None):
        self.paths: "List[str]" = list(dict.fromkeys(paths))
        self.index_of: "Dict[str, int]" = {}
        self._sim_view_factory = sim_view_factory or _simulation_view
        self._logger = logger or _LOG
        self._sim_view = None
        self._view = None
        self._rows: "Optional[np.ndarray]" = None          # rows[i] = row of self.paths[i] in the view buffers
        self._xf = None                                      # (count, 7) float32: x y z qx qy qz qw (view buffer)
        self._vel = None                                     # (count, 6) float32: v w (view buffer)
        self._sampled_step = None
        self._ok = False
        self._bind_failures = 0
        self.generation = 0                                  # bumps on every (re)bind: consumers refresh cached rows

    # -- binding -------------------------------------------------------------
    @property
    def bound(self) -> bool:
        return self._view is not None

    def bind(self) -> bool:
        """Create the RigidBodyView on the current SimulationView. False when physics is not up yet."""
        sim_view = self._sim_view_factory()
        if sim_view is None or not self.paths:
            self._drop()
            return False
        try:
            view = sim_view.create_rigid_body_view(list(self.paths))
            prim_paths = [str(p) for p in view.prim_paths]
            count = int(view.count)
        except Exception as exc:
            self._drop()
            self._bind_failures += 1
            if self._bind_failures in (1, 10, 100):
                self._logger.warning("[LINK VIEW] bind failed (%s): %s: %s", self._bind_failures, type(exc).__name__, exc)
            return False
        if count != len(prim_paths):
            prim_paths = prim_paths[:count]
        row_of = {p: i for i, p in enumerate(prim_paths)}
        missing = [p for p in self.paths if p not in row_of]
        if missing:
            self._logger.warning("[LINK VIEW] %d tracked prim(s) missing from the PhysX view (not rigid bodies?): %s",
                                 len(missing), missing[:6])
        self.paths = [p for p in self.paths if p in row_of]
        self.index_of = {p: i for i, p in enumerate(self.paths)}
        self._rows = np.asarray([row_of[p] for p in self.paths], dtype=np.int64)
        self._sim_view = sim_view
        self._view = view
        self._xf = self._vel = None
        self._sampled_step = None
        self._bind_failures = 0
        self.generation += 1
        self._logger.info("[LINK VIEW] bound %d/%d rigid bodies in one tensor view (generation %d)",
                          len(self.paths), count, self.generation)
        return True

    def _drop(self) -> None:
        self._view = None
        self._sim_view = None
        self._xf = self._vel = None
        self._ok = False

    def _needs_rebind(self) -> bool:
        if self._view is None:
            return True
        sim_view = self._sim_view_factory()
        if sim_view is None or sim_view is not self._sim_view:
            return True
        valid = getattr(sim_view, "is_valid", True)
        return not bool(valid)

    # -- per step --------------------------------------------------------------
    def sample(self, step_id) -> bool:
        """Read every tracked body's transform + velocity once for `step_id` (repeat calls are free)."""
        if step_id == self._sampled_step:
            return self._ok
        if self._needs_rebind() and not self.bind():
            self._sampled_step = step_id
            self._ok = False
            return False
        self._sampled_step = step_id            # after bind(): a (re)bind clears the guard
        try:
            xf = self._view.get_transforms()
            vel = self._view.get_velocities()
        except Exception as exc:
            self._logger.warning("[LINK VIEW] sample failed, rebinding next step: %s: %s", type(exc).__name__, exc)
            self._drop()
            return False
        self._xf = np.asarray(xf)[self._rows]
        self._vel = np.asarray(vel)[self._rows]
        self._ok = True
        return True

    @property
    def ok(self) -> bool:
        return self._ok

    def transforms(self) -> "Optional[np.ndarray]":
        """(n, 7) float32 world transforms, x y z qx qy qz qw, in self.paths order (valid until the next sample)."""
        return self._xf if self._ok else None

    def velocities(self) -> "Optional[np.ndarray]":
        """(n, 6) float32 world linear (COM) + angular velocities, self.paths order."""
        return self._vel if self._ok else None

    def row(self, path: str) -> int:
        """Row of `path` in transforms()/velocities(), -1 when the path is not tracked."""
        return self.index_of.get(path, -1)

    def pose_wxyz(self, row: int) -> "Tuple[np.ndarray, np.ndarray]":
        """(position (3,), quaternion (4,) w x y z) float64 — the layout Isaac's RigidPrim.get_world_poses returns."""
        t = self._xf[row]
        return (np.array((t[0], t[1], t[2]), dtype=np.float64),
                np.array((t[6], t[3], t[4], t[5]), dtype=np.float64))

    def velocity(self, row: int) -> np.ndarray:
        """(6,) float64 world velocity: linear (COM) then angular — RigidPrim.get_velocities layout."""
        return np.asarray(self._vel[row], dtype=np.float64)
