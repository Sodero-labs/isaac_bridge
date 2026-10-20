"""Tracked free objects (a ball, a goal, boxes ...): world pose and velocity publisher.

A prim tagged in the world USD with

    custom bool   rc:trackedObject = true
    custom string rc:objectKind    = "ball" | "goal" | ...
    custom double rc:objectRadius / rc:objectWidth / ...   (optional, forwarded as-is)

is published on the state port as  b'TM/<prim name>/object_state' + b'||' + <JSON>  next to the robots'
sim_state / imu_state frames (same PUB, different topic suffix; consumers filter by topic). Dynamic objects
(UsdPhysics RigidBodyAPI) are read from the PhysX rigid-body view and published at OBJECT_PUB_HZ; static ones
(colliders only, e.g. a goal) are read from USD once and repeated at STATIC_PUB_HZ so a late subscriber still
learns where they are:

    {"name": "soccer_ball", "kind": "ball", "static": false, "stamp_ns": <sim time>, "seq": n,
     "position": [x, y, z], "orientation_xyzw": [x, y, z, w],
     "linear_velocity": [vx, vy, vz], "angular_velocity": [wx, wy, wz],
     "home": [x, y, z]  (world position when the object was bound, i.e. its spawn position),
     "radius": 0.045, ...other rc:object* attributes}

Inbound (command port, multipart [b'TM/', <name>, b'object_teleport', <JSON>]):
    {"position": [x, y, z], "orientation_xyzw": [x, y, z, w] (optional), "zero_velocity": true}
moves a dynamic object (for instance back home after it rolled away).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import omni.usd
from isaacsim.core.prims import RigidPrim

from .device import WrappedDevice

if TYPE_CHECKING:
    from typing import List, Tuple
    from isaacsim.core.api import SimulationContext
    from pxr.Usd import Prim

OBJECT_STATE_TOPIC = "object_state"
TELEPORT_TYPE = "object_teleport"
OBJECT_PUB_HZ = 50.0        # dynamic objects (simulation time)
STATIC_PUB_HZ = 2.0         # static objects (the pose never changes; repeated for late subscribers)

_LOG = logging.getLogger("isaac_bridge")


def _attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.IsValid() else None


class WrappedRigidObject(WrappedDevice):
    """One tracked object. Dynamic: PhysX rigid-body view (needs a valid physics view, else the moderator
    retries). Static: USD world transform, sampled once."""

    def __new__(cls, sim_context: "SimulationContext", path: str, name: str):
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path) if stage else None
        if prim is None or not prim.IsValid():
            return None
        dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI) and not bool(_attr(prim, "physics:kinematicEnabled"))
        rigid = None
        if dynamic:
            rigid = RigidPrim(path, name=f"{name}_obj")
            if not rigid.is_physics_handle_valid():
                return None          # physics view not live yet: the moderator retries
        inst = super().__new__(cls)
        inst._sim_context = sim_context
        inst._name = name
        inst._path = path
        inst._dynamic = bool(dynamic)
        inst._rigid = rigid
        return inst

    def __init__(self, *args, **kwargs):
        self._logger = _LOG
        prim = self._prim()
        self._kind = str(_attr(prim, "rc:objectKind") or "object") if prim else "object"
        self._extra = {}
        if prim:
            for a in prim.GetAttributes():
                n = a.GetName()
                if n.startswith("rc:object") and n not in ("rc:objectKind",):
                    key = n[len("rc:object"):]
                    key = key[0].lower() + key[1:]
                    try:
                        self._extra[key] = float(a.Get())
                    except (TypeError, ValueError):
                        self._extra[key] = a.Get()
        self._seq = 0
        self._state = None          # (pos3, quat_xyzw4, v3, w3) float64
        self._last_pub_t = None
        self._link_view = None
        self._view_row = -1
        self._view_gen = -1
        self._pending_teleport = None
        self._logged = False
        self._home = self._usd_world_pose()      # spawn pose: bound before physics moved the object
        if not self._dynamic:
            self._sample_static()

    # -- WrappedDevice ------------------------------------------------------
    def destroy(self) -> None:
        self._rigid = None

    @classmethod
    def is_tracked_object(cls, prim: "Prim") -> bool:
        try:
            return prim.HasAttribute("rc:trackedObject") and bool(prim.GetAttribute("rc:trackedObject").Get())
        except Exception:
            return False

    def update_target(self, data_type: str, data: bytes) -> None:
        if data_type != TELEPORT_TYPE:
            return
        try:
            self._pending_teleport = json.loads(bytes(data).decode("utf-8"))
        except Exception as e:
            self._logger.warning(f"[{self._name}] bad {TELEPORT_TYPE} payload: {e}")

    def set_link_view(self, view) -> None:
        """The stage-wide rigid-body view (link_view.LinkStateView); None = RigidPrim reads."""
        self._link_view = view
        self._view_row = -1
        self._view_gen = -1

    def _publish_due(self) -> bool:
        try:
            t = float(self._sim_context.current_time)
        except Exception:
            t = 0.0
        period = 1.0 / (OBJECT_PUB_HZ if self._dynamic else STATIC_PUB_HZ)
        return not (self._last_pub_t is not None and 0.0 <= (t - self._last_pub_t) < period)

    def synchronize(self) -> None:
        if not self._dynamic:
            return
        if self._pending_teleport is not None:
            tp, self._pending_teleport = self._pending_teleport, None
            self._apply_teleport(tp)
        elif not self._publish_due():
            return                      # sampled only on the steps that publish, not on every physics step
        view = self._link_view
        try:
            if view is not None and view.ok:
                if self._view_gen != view.generation:
                    self._view_row = view.row(self._path)
                    self._view_gen = view.generation
                if self._view_row >= 0:
                    p, q_wxyz = view.pose_wxyz(self._view_row)
                    v6 = view.velocity(self._view_row)
                else:
                    view = None
            if view is None or not view.ok:
                pos, quat = self._rigid.get_world_poses()
                vel = self._rigid.get_velocities()
                p = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
                q_wxyz = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
                v6 = np.asarray(vel, dtype=np.float64).reshape(-1)
        except Exception as e:
            self._logger.debug(f"[{self._name}] object sample failed: {e}")
            return
        self._state = (p, np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]), v6[:3], v6[3:6])
        if not self._logged:
            self._logged = True
            self._logger.info(f"[{self._name}] tracked {self._kind} (dynamic) at {np.round(p, 3).tolist()} {self._extra}")

    def get_actual(self) -> "List[Tuple[str, bytes]]":
        if self._state is None:
            return []
        try:
            t = float(self._sim_context.current_time)
        except Exception:
            t = 0.0
        period = 1.0 / (OBJECT_PUB_HZ if self._dynamic else STATIC_PUB_HZ)
        if self._last_pub_t is not None and 0.0 <= (t - self._last_pub_t) < period:
            return []
        self._last_pub_t = t
        self._seq += 1
        p, q, v, w = self._state
        msg = {"name": self._name, "kind": self._kind, "static": not self._dynamic,
               "stamp_ns": int(t * 1e9), "seq": self._seq,
               "position": [round(float(x), 5) for x in p], "orientation_xyzw": [round(float(x), 6) for x in q],
               "linear_velocity": [round(float(x), 4) for x in v], "angular_velocity": [round(float(x), 4) for x in w]}
        if self._home is not None:
            msg["home"] = [round(float(x), 5) for x in self._home[0]]
        msg.update(self._extra)
        return [(OBJECT_STATE_TOPIC, json.dumps(msg, separators=(",", ":")).encode("utf-8"))]

    # -- helpers ------------------------------------------------------------
    def _prim(self):
        try:
            stage = omni.usd.get_context().get_stage()
            return stage.GetPrimAtPath(self._path) if stage else None
        except Exception:
            return None

    def _usd_world_pose(self):
        """(pos3, quat_xyzw4) of the prim from USD (composed world transform), or None."""
        prim = self._prim()
        if not prim:
            return None
        try:
            from pxr import Usd, UsdGeom
            m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            tr = m.ExtractTranslation()
            rq = m.ExtractRotationQuat()
            im = rq.GetImaginary()
            return (np.array([tr[0], tr[1], tr[2]], dtype=np.float64),
                    np.array([im[0], im[1], im[2], rq.GetReal()], dtype=np.float64))
        except Exception as e:
            self._logger.warning(f"[{self._name}] USD pose read failed: {e}")
            return None

    def _sample_static(self) -> None:
        pose = self._usd_world_pose()
        if pose is None:
            return
        self._state = (pose[0], pose[1], np.zeros(3), np.zeros(3))
        self._logger.info(f"[{self._name}] tracked {self._kind} (static) at {np.round(self._state[0], 3).tolist()} {self._extra}")

    def _apply_teleport(self, tp: dict) -> None:
        try:
            pos = np.asarray(tp.get("position"), dtype=np.float64).reshape(1, 3)
            q = tp.get("orientation_xyzw")
            ori = None
            if q is not None:
                x, y, z, w = (float(v) for v in q)
                ori = np.array([[w, x, y, z]], dtype=np.float64)
            self._rigid.set_world_poses(positions=pos, orientations=ori)
            if tp.get("zero_velocity", True):
                self._rigid.set_velocities(np.zeros((1, 6), dtype=np.float64))
            self._logger.info(f"[{self._name}] teleported to {pos.reshape(-1).tolist()}")
        except Exception as e:
            self._logger.warning(f"[{self._name}] teleport failed: {e}")
