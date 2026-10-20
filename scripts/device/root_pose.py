"""Articulation root pose on the state port: b'TM/<robot>/root_pose' + b'||' + <JSON>, at most 20 Hz.

sim_state carries joints only and imu_state only the IMU point of robots that author rc:imuLink, so this frame is
how an observer learns where a robot is in its world — whether it fell, how fast it drives. Every articulation
publishes its PhysX root link next to its sim_state (same PUB socket, own topic suffix; consumers filter by topic):

    {"pos": [x, y, z], "quat_wxyz": [w, x, y, z], "lin_vel": [vx, vy, vz], "ang_vel": [wx, wy, wz],
     "sim_time": <s>, "seq": n}

World frame. pos / quat_wxyz = the root link frame (origin and orientation); lin_vel = the root link's
centre-of-mass linear velocity and ang_vel its angular velocity, as PhysX reports them (the convention of
object_state's linear_velocity and of Isaac's get_velocities). One frame per articulation, carrying its root link
only.

Rate: on simulation time like object_state: a frame goes out when at least 1/ROOT_POSE_PUB_HZ of simulation time
passed since the last one (a timeline reset, which sends the clock backwards, publishes at once).
RC_ISAAC_ROOT_POSE_HZ lowers the rate (0 = off) and never raises it. The pose is read only on the steps that publish,
from buffers the step has already sampled (link_view.LinkStateView), so in the normal case this adds no backend
call to a physics step.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import config

ROOT_POSE_TOPIC = "root_pose"
ROOT_POSE_PUB_HZ = 20.0          # ceiling: observers poll where robots are, they do not control them
ROOT_POSE_KEYS = ("pos", "quat_wxyz", "lin_vel", "ang_vel", "sim_time")
_LENGTHS = {"pos": 3, "quat_wxyz": 4, "lin_vel": 3, "ang_vel": 3}


def root_pose_hz_from_env(environ=None) -> float:
    """RC_ISAAC_ROOT_POSE_HZ clamped to [0, ROOT_POSE_PUB_HZ]; unset/invalid -> ROOT_POSE_PUB_HZ, <= 0 -> 0 (off)."""
    raw = str((os.environ if environ is None else environ).get(config.ROOT_POSE_HZ_ENV, "")).strip()
    try:
        hz = float(raw) if raw else ROOT_POSE_PUB_HZ
    except ValueError:
        return ROOT_POSE_PUB_HZ
    if not np.isfinite(hz) or hz <= 0.0:
        return 0.0
    return min(hz, ROOT_POSE_PUB_HZ)


class RootPoseRate:
    """Sim-time rate limit of one robot's root_pose frames (the object_state rule)."""

    __slots__ = ("period", "last_t")

    def __init__(self, hz: float = ROOT_POSE_PUB_HZ):
        hz = min(float(hz), ROOT_POSE_PUB_HZ) if hz and hz > 0.0 else 0.0
        self.period = (1.0 / hz) if hz > 0.0 else None
        self.last_t = None

    @property
    def enabled(self) -> bool:
        return self.period is not None

    def due(self, t: float) -> bool:
        """True when a frame may go out at sim time `t`. Only mark() moves the clock: a step that had no pose to
        publish leaves the next step due."""
        if self.period is None:
            return False
        last = self.last_t
        return not (last is not None and 0.0 <= (t - last) < self.period)

    def mark(self, t: float) -> None:
        self.last_t = t


def encode_root_pose(sim_time: float, seq: int, pos, quat_wxyz, lin_vel, ang_vel):
    """The JSON payload (bytes), or None when a value is missing, the wrong size or not finite (a robot whose physics
    blew up publishes nothing rather than NaN, which strict JSON parsers reject)."""
    try:
        parts = [np.asarray(v, dtype=np.float64).reshape(-1) for v in (pos, quat_wxyz, lin_vel, ang_vel)]
        t = float(sim_time)
    except (TypeError, ValueError):
        return None
    if [p.size for p in parts] != [3, 4, 3, 3] or not np.isfinite(t):
        return None
    if not all(np.isfinite(p).all() for p in parts):
        return None
    p, q, v, w = parts
    msg = {"pos": [round(float(x), 5) for x in p], "quat_wxyz": [round(float(x), 6) for x in q],
           "lin_vel": [round(float(x), 4) for x in v], "ang_vel": [round(float(x), 4) for x in w],
           "sim_time": round(t, 6), "seq": int(seq)}
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode_root_pose(payload: bytes) -> dict:
    """Reference decoder (what a subscriber needs): the payload as a dict, ValueError when it is not a root_pose."""
    try:
        msg = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"root_pose payload is not JSON: {exc}") from None
    if not isinstance(msg, dict) or any(k not in msg for k in ROOT_POSE_KEYS):
        raise ValueError("root_pose payload lacks " + ", ".join(k for k in ROOT_POSE_KEYS if not isinstance(msg, dict) or k not in msg))
    for key, n in _LENGTHS.items():
        value = msg[key]
        if not isinstance(value, list) or len(value) != n or not all(isinstance(x, (int, float)) for x in value):
            raise ValueError(f"root_pose {key} must be {n} numbers")
    if not isinstance(msg["sim_time"], (int, float)):
        raise ValueError("root_pose sim_time must be a number")
    return msg
