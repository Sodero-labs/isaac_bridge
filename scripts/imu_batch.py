"""Every robot's emulated IMU computed in one vectorized pass.

The IMU of one robot is about 25 numpy operations on 3-element arrays (a quaternion to matrix, two cross
products, a first-order filter, the foot-contact heights), where the call overhead dwarfs the arithmetic. The
same operations on (N, 3) arrays cost about as much for N robots, and the pass runs on every physics step.

The per-robot path in device/articulation.py stays as the fallback (a robot the link view does not carry, or a
failed batch read). This module owns the filter state of the robots it computes, so a robot that moves between
the two paths restarts its accelerometer filter.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from . import protocol

if TYPE_CHECKING:
    from typing import Sequence

_LOG = logging.getLogger("isaac_bridge")

GRAVITY = np.array([0.0, 0.0, -9.81])


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions w x y z -> (..., 3, 3) rotation matrices (column vectors: v' = R @ v)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((
        np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1),
        np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1),
        np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1),
    ), axis=-2)


class ImuBatch:
    """Bound to a set of (device, imu binding) pairs; `compute()` hands every device its packed imu_state."""

    def __init__(self, bindings: "Sequence[tuple]", *, logger=None):
        """bindings: [(device, spec)], spec as WrappedArticulation.imu_binding() returns it:
        {"path", "contact_paths", "offset" (3,), "com_local" (3,), "contact_offsets" [(3,)], "contact_height",
         "ground_z", "accel_tau"}."""
        self._logger = logger or _LOG
        self.devices = [d for d, _ in bindings]
        specs = [s for _, s in bindings]
        n = len(specs)
        self.count = n
        self.paths = [s["path"] for s in specs]
        self.contact_paths = [list(s.get("contact_paths") or []) for s in specs]
        self.max_contacts = max((len(c) for c in self.contact_paths), default=0)
        self.offsets = np.array([np.asarray(s.get("offset", (0, 0, 0)), dtype=np.float64) for s in specs]).reshape(n, 3) if n else np.zeros((0, 3))
        com_local = np.array([s.get("com_local", (0, 0, 0)) for s in specs], dtype=np.float64).reshape(n, 3)
        self.velocity_offsets = self.offsets - com_local
        self.contact_height = np.array([float(s.get("contact_height", 0.01)) for s in specs], dtype=np.float64)
        self.ground_z = np.array([float(s.get("ground_z", 0.0)) for s in specs], dtype=np.float64)
        self.accel_tau = np.array([float(s.get("accel_tau", 0.02)) for s in specs], dtype=np.float64)
        # (n, max_contacts, 3) sole offsets, padded with zeros for robots with fewer contacts
        self.contact_offsets = np.zeros((n, self.max_contacts, 3), dtype=np.float64)
        for i, s in enumerate(specs):
            for j, off in enumerate(s.get("contact_offsets") or []):
                if j < self.max_contacts:
                    self.contact_offsets[i, j] = np.asarray(off, dtype=np.float64)
        self._n_contacts = np.array([len(c) for c in self.contact_paths], dtype=np.int64)

        self._rows = None                  # (n,) row of each IMU link in the link view
        self._contact_rows = None          # (n, max_contacts)
        self._rows_gen = -1
        self._prev_v = None                # (n, 3) IMU-point velocity of the previous computed step
        self._prev_t = None                # sim time of that sample
        self._accel_filt = None            # (n, 3) filtered proper acceleration
        self._seq = np.zeros(n, dtype=np.int64)
        # One ImuState record per robot, filled column-wise and sliced into per-robot bytes: the wire record has
        # a fixed layout, so N packs of 8 fields each become 8 assignments plus one tobytes.
        self._rec = np.zeros(n, dtype=protocol._IMU_DTYPE)
        self._rec_size = int(self._rec.itemsize)

    # -- binding ---------------------------------------------------------------
    def _resolve_rows(self, view) -> bool:
        if self._rows_gen == view.generation and self._rows is not None:
            return True
        rows = np.array([view.row(p) for p in self.paths], dtype=np.int64)
        contact_rows = np.full((self.count, self.max_contacts), -1, dtype=np.int64)
        for i, paths in enumerate(self.contact_paths):
            for j, p in enumerate(paths[:self.max_contacts]):
                contact_rows[i, j] = view.row(p)
        ok = bool(self.count) and bool((rows >= 0).all())
        for i in range(self.count):
            if not ok:
                break
            if (contact_rows[i, :self._n_contacts[i]] < 0).any():
                ok = False
        if not ok:
            missing = [self.paths[i] for i in range(self.count) if rows[i] < 0]
            self._logger.info("[IMU BATCH] %d link(s) not in the link view (%s...); per-robot IMU",
                              len(missing), missing[:2])
            self._rows = None
            self._rows_gen = view.generation
            return False
        self._rows = rows
        self._contact_rows = contact_rows
        self._rows_gen = view.generation
        self._prev_v = None
        self._prev_t = None
        self._accel_filt = None
        self._logger.info("[IMU BATCH] %d robot(s) computed in one pass (%d contact links each at most)",
                          self.count, self.max_contacts)
        return True

    # -- per step ----------------------------------------------------------------
    def compute(self, view, sim_time: float) -> bool:
        """One vectorized IMU pass for every bound robot; each device gets its packed payload. False = the
        caller should fall back to the per-robot path this step."""
        if not self.count or view is None or not view.ok:
            return False
        if not self._resolve_rows(view):
            return False
        xf = view.transforms()
        vel = view.velocities()
        if xf is None or vel is None:
            return False
        try:
            body = np.asarray(xf)[self._rows]                    # (n, 7) x y z qx qy qz qw
            v6 = np.asarray(vel)[self._rows].astype(np.float64)  # (n, 6)
            p = body[:, :3].astype(np.float64)
            q = body[:, [6, 3, 4, 5]].astype(np.float64)         # -> w x y z
            R = quat_to_matrix_wxyz(q)                           # (n, 3, 3)
            v_w, w_w = v6[:, :3], v6[:, 3:6]

            r_w = np.einsum("nij,nj->ni", R, self.offsets)        # IMU offset rotated into the world
            p_imu = p + r_w
            r_from_com = np.einsum("nij,nj->ni", R, self.velocity_offsets)
            v_imu = v_w + np.cross(w_w, r_from_com)

            a_w = np.zeros_like(v_imu)
            dt = 0.0
            if self._prev_v is not None and self._prev_t is not None:
                dt = float(sim_time - self._prev_t)
                if dt > 1e-6:
                    a_w = (v_imu - self._prev_v) / dt
            self._prev_v, self._prev_t = v_imu, sim_time

            # first-order accelerometer filter, per robot time constant (rc:imuAccelTau)
            if self._accel_filt is None or dt <= 0.0:
                a_f = a_w.copy()
            else:
                alpha = (dt / (self.accel_tau + dt))[:, None]
                a_f = self._accel_filt + alpha * (a_w - self._accel_filt)
            self._accel_filt = a_f

            accel_b = np.einsum("nji,nj->ni", R, a_f - GRAVITY)   # R^T (a - g): proper acceleration, IMU frame
            gyro_b = np.einsum("nji,nj->ni", R, w_w)

            contacts = np.zeros((self.count, self.max_contacts), dtype=np.uint8)
            if self.max_contacts:
                rows = np.maximum(self._contact_rows, 0)
                cb = np.asarray(xf)[rows.reshape(-1)].reshape(self.count, self.max_contacts, 7)
                cp = cb[:, :, :3].astype(np.float64)
                cq = cb[:, :, [6, 3, 4, 5]].astype(np.float64)
                cR = quat_to_matrix_wxyz(cq.reshape(-1, 4)).reshape(self.count, self.max_contacts, 3, 3)
                sole = cp + np.einsum("ncij,ncj->nci", cR, self.contact_offsets)
                touching = (sole[:, :, 2] - self.ground_z[:, None]) < self.contact_height[:, None]
                valid = self._contact_rows >= 0
                contacts = (touching & valid).astype(np.uint8)
        except Exception as exc:
            self._logger.warning("[IMU BATCH] pass failed, per-robot IMU this step: %s: %s",
                                 type(exc).__name__, exc)
            return False

        self._publish(sim_time, q[:, [1, 2, 3, 0]], gyro_b, accel_b, v_imu, p_imu, contacts)
        return True

    def _publish(self, sim_time, quat_xyzw, gyro_b, accel_b, v_imu, p_imu, contacts) -> None:
        self._seq += 1
        rec = self._rec
        rec["sim_time_ns"] = int(sim_time * 1e9)
        rec["seq"] = self._seq
        rec["quat_xyzw"] = quat_xyzw
        rec["gyro"] = gyro_b
        rec["accel"] = accel_b
        rec["lin_vel_w"] = v_imu
        rec["pos_w"] = p_imu
        rec["contacts"] = 0
        if self.max_contacts:
            k = min(self.max_contacts, rec["contacts"].shape[1])
            rec["contacts"][:, :k] = contacts[:, :k]
        blob = rec.tobytes()            # the same bytes as one ImuPacker.pack() per robot
        size = self._rec_size
        for i, device in enumerate(self.devices):
            device.set_imu_blob(blob[i * size:(i + 1) * size], int(self._seq[i]))
