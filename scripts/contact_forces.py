"""Measured contact loads of opted-in robot links, one PhysX tensor read for the whole stage per step.

A robot opts in with `rc:contactForceLinks` (link names under <robot>/links/, at most 8). A controller can then
make support decisions on how much weight each link carries — lift a foot only once it is unloaded, release an arm
only once the feet carry the body — which imu_state's two geometric foot flags cannot tell.

PhysX reports contacts only for rigid bodies that carry PhysxContactReportAPI when the scene is parsed (a
timeline play). The world USD should author it; a robot whose links lack it gets the API authored at bind time
and publishes zeros until the next play (logged).

Like link_view.LinkStateView, the RigidContactView is (re)bound lazily: every timeline play hands out a new
SimulationView and invalidates the old tensor views.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from . import protocol
from .link_view import _simulation_view

if TYPE_CHECKING:
    from typing import List, Sequence

_LOG = logging.getLogger("isaac_bridge")


class ContactForceBatch:
    """Bound to [(device, [link prim paths])]; `compute()` hands every device its packed contact_forces."""

    def __init__(self, bindings: "Sequence[tuple]", *, sim_view_factory=None, logger=None):
        self._logger = logger or _LOG
        self.devices = [d for d, _ in bindings]
        self.link_paths: "List[List[str]]" = [list(paths)[:protocol.CONTACT_FORCES_MAX] for _, paths in bindings]
        self.paths = [p for paths in self.link_paths for p in paths]
        self._sim_view_factory = sim_view_factory or _simulation_view
        self._sim_view = None
        self._view = None
        self._rows = None                     # (len(self.paths),) row of each path in the view's force buffer
        self._seq = np.zeros(len(self.devices), dtype=np.int64)
        self._bind_failures = 0

    def _bind(self) -> bool:
        sim_view = self._sim_view_factory()
        if sim_view is None or not self.paths:
            self._view = self._sim_view = None
            return False
        if self._view is not None and sim_view is self._sim_view and bool(getattr(sim_view, "is_valid", True)):
            return True
        try:
            view = sim_view.create_rigid_contact_view(list(self.paths))
            sensors = [str(p) for p in getattr(view, "sensor_paths", self.paths)]
        except Exception as exc:
            self._view = self._sim_view = None
            self._bind_failures += 1
            if self._bind_failures in (1, 10, 100):
                self._logger.warning("[CONTACT FORCES] bind failed (%s): %s: %s", self._bind_failures,
                                     type(exc).__name__, exc)
            return False
        row_of = {p: i for i, p in enumerate(sensors)}
        missing = [p for p in self.paths if p not in row_of]
        if missing:
            self._logger.warning("[CONTACT FORCES] %d link(s) missing from the contact view: %s", len(missing), missing[:4])
            self._view = self._sim_view = None
            return False
        self._rows = np.asarray([row_of[p] for p in self.paths], dtype=np.int64)
        self._view, self._sim_view = view, sim_view
        self._bind_failures = 0
        self._logger.info("[CONTACT FORCES] bound %d link(s) of %d robot(s) in one contact view",
                          len(self.paths), len(self.devices))
        return True

    def compute(self, sim_time: float, dt: float) -> bool:
        """Read net contact forces once and give each device its payload. False = nothing published this step."""
        if dt <= 0 or not self._bind():
            return False
        try:
            forces = np.asarray(self._view.get_net_contact_forces(dt), dtype=np.float64).reshape(-1, 3)
        except Exception as exc:
            self._logger.warning("[CONTACT FORCES] read failed, rebinding next step: %s: %s", type(exc).__name__, exc)
            self._view = self._sim_view = None
            return False
        normal_z = forces[self._rows, 2]
        stamp = int(sim_time * 1e9)
        self._seq += 1
        start = 0
        for i, (device, paths) in enumerate(zip(self.devices, self.link_paths)):
            device.set_contact_force_payload(protocol.pack_contact_forces(stamp, int(self._seq[i]),
                                                                          normal_z[start:start + len(paths)]))
            start += len(paths)
        return True
