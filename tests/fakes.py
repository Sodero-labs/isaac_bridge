"""Stand-ins for the Isaac Sim objects the bridge talks to, and builders for bound devices, shared by the tests.

conftest.py stubs the Isaac Sim modules themselves; these fakes give the devices something that behaves.
"""
from __future__ import annotations

import collections
import logging

import numpy as np

from scripts import protocol
from scripts.device import articulation, rigid_object
from scripts.link_view import LinkStateView
from scripts.moderator import moderator as moderator_mod


def rand_quat(rng) -> np.ndarray:
    """A random unit quaternion w x y z with w >= 0."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


class FakeSimContext:
    """The parts of Isaac Sim's SimulationContext the bridge reads."""

    def __init__(self, current_time: float = 0.0, physics_dt: float = 1 / 250, playing: bool = True):
        self.current_time = current_time
        self._physics_dt = physics_dt
        self._playing = playing

    def get_physics_dt(self) -> float:
        return self._physics_dt

    def is_playing(self) -> bool:
        return self._playing


# ------------------------------------------------------------------ articulations
class FakePhysicsView:
    """omni.physics.tensors ArticulationView over one articulation (numpy frontend). `calls` records "get_q" /
    "get_dq" for the reads and (tensor, data) for the writes, tensor in pos / vel / eff / kp / kd."""

    def __init__(self, n_dof: int, q=None):
        self.n = n_dof
        self.q = np.zeros((1, n_dof), dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32).reshape(1, n_dof)
        self.dq = np.zeros((1, n_dof), dtype=np.float32)
        self.calls = []

    def get_dof_positions(self):
        self.calls.append("get_q")
        return self.q

    def get_dof_velocities(self):
        self.calls.append("get_dq")
        return self.dq

    def _write(self, tensor, data, indices):
        assert data.dtype == np.float32 and data.shape == (1, self.n) and indices.dtype == np.uint32
        self.calls.append((tensor, data.copy()))

    def set_dof_position_targets(self, data, indices):
        self._write("pos", data, indices)

    def set_dof_velocity_targets(self, data, indices):
        self._write("vel", data, indices)

    def set_dof_actuation_forces(self, data, indices):
        self._write("eff", data, indices)

    def set_dof_stiffnesses(self, data, indices):
        self._write("kp", data, indices)

    def set_dof_dampings(self, data, indices):
        self._write("kd", data, indices)

    def writes(self):
        """The tensors written, in order."""
        return [c[0] for c in self.calls if isinstance(c, tuple)]

    def last(self, tensor):
        """The last array written to `tensor`."""
        return [c[1] for c in self.calls if isinstance(c, tuple) and c[0] == tensor][-1]


class FakeArticulation:
    """Isaac Sim's Articulation wrapper over a FakePhysicsView. Its own joint reads must not be needed while the raw
    view works."""

    def __init__(self, view, names):
        self._physics_view = view
        self.dof_names = list(names)
        self.num_dof = len(names)

    def get_joint_positions(self):
        raise AssertionError("the wrapper path must not be used when the raw view works")

    get_joint_velocities = get_joint_positions

    def get_measured_joint_efforts(self):
        return np.zeros((1, self.num_dof), dtype=np.float32)


def make_robot(names=3, *, path: str = "/W/r", q=None, sim_time: float = 0.5, physics_dt: float = 1 / 250):
    """A WrappedArticulation bound to fakes: raw tensor view, actuator order = DOF order, USD metadata already read
    (none), no respawn pending. `names` is the DOF names or their count. Returns (robot, its FakePhysicsView)."""
    names = [f"j{i}" for i in range(names)] if isinstance(names, int) else list(names)
    n = len(names)
    a = object.__new__(articulation.WrappedArticulation)       # __new__ would bind a real Articulation
    a._sim_context = FakeSimContext(current_time=sim_time, physics_dt=physics_dt)
    a._name, a._path = path.rsplit("/", 1)[-1], path
    pv = FakePhysicsView(n, q=q)
    a._art = FakeArticulation(pv, names)
    a._n = n
    a.__init__()
    a._meta_loaded = True
    a._imu = False
    a._perm_arr = np.arange(n)
    a._n_dof = a._n_dof_perm = n
    a._passive_kd = np.zeros(n)
    a._logged_apply = True
    a._fmax_canon = False                  # no drive force limit, unless a test looks one up
    a._respawn_left = 0
    a._usd_joint_info = lambda: [(nm, False, None, 0.0) for nm in names]
    return a, pv


def command(n: int, q_d=None, *, mode: int = protocol.MODE_POSITION, modes=None, dq_d=None, tau_d=None, kp=None,
            kd=None) -> bytes:
    """A SimCommand payload for the window [0, n): every DOF in `mode` (or the per-DOF `modes`), the rest IDLE."""
    M = protocol.SIM_MAX_JOINTS
    m = np.full(M, protocol.MODE_IDLE, dtype=np.uint8)
    m[:n] = mode if modes is None else modes
    d = np.zeros(5 * M)
    for i, values in enumerate((q_d, dq_d, tau_d, kp, kd)):
        if values is not None:
            d[i * M:i * M + n] = values
    return protocol._CMD.pack(1, 1, n, *m.tolist(), *d.tolist())


# ------------------------------------------------------------------ rigid bodies
class FakeRigidBodyView:
    def __init__(self, paths, store):
        self.prim_paths = list(paths)
        self.count = len(paths)
        self._store = store
        self.reads = 0

    def get_transforms(self):
        self.reads += 1
        return np.array([self._store[p][0] for p in self.prim_paths], dtype=np.float32)

    def get_velocities(self):
        return np.array([self._store[p][1] for p in self.prim_paths], dtype=np.float32)


class FakeSimView:
    """omni.physics.tensors SimulationView: knows the rigid bodies of `store` (path -> (x y z qx qy qz qw, v6)) and
    hands them out in its own `order`."""

    def __init__(self, store, order=None):
        self.store = store
        self.order = order
        self.is_valid = True
        self.created = 0

    def create_rigid_body_view(self, paths):
        self.created += 1
        return FakeRigidBodyView([p for p in (self.order or list(self.store)) if p in paths], self.store)


def random_store(paths, rng):
    """path -> (random transform x y z qx qy qz qw, random velocity) for FakeSimView."""
    return {p: (np.concatenate([rng.standard_normal(3), rand_quat(rng)[[1, 2, 3, 0]]]), rng.standard_normal(6))
            for p in paths}


def sampled_link_view(store, paths=None, step: int = 1) -> LinkStateView:
    """A LinkStateView over `store` on one simulation view (as during a play), sampled once."""
    paths = list(store) if paths is None else paths
    sim_view = FakeSimView(store)
    view = LinkStateView(paths, sim_view_factory=lambda: sim_view, logger=logging.getLogger("isaac_bridge"))
    assert view.sample(step)
    return view


def make_tracked_object(name: str = "ball", path: str = "/W/ball", link_view=None):
    """A dynamic WrappedRigidObject read through `link_view` (it has no RigidPrim to fall back on)."""
    obj = object.__new__(rigid_object.WrappedRigidObject)
    obj._logger = logging.getLogger("isaac_bridge")
    obj._name, obj._path, obj._dynamic, obj._kind = name, path, True, "ball"
    obj._extra, obj._seq, obj._state, obj._last_pub_t = {}, 0, None, None
    obj._pending_teleport, obj._logged, obj._home = None, True, None
    obj._sim_context = FakeSimContext(current_time=0.0)
    obj._rigid = None
    obj.set_link_view(link_view)
    return obj


# ------------------------------------------------------------------ moderator
class FakeHandler:
    """MessageHandler without sockets: `inbound` frames are drained as usual, published frames land in `sent`."""

    def __init__(self):
        self.inbound = collections.deque()
        self.sent = []
        self.is_server_running = True

    def drain_inbound(self, handler, limit=None) -> int:
        n = 0
        while self.inbound:
            handler(self.inbound.popleft())
            n += 1
        return n

    def send(self, name, frames) -> None:
        self.sent.append((name, list(frames)))

    def start_server(self) -> bool:
        return True

    def stop_server(self) -> bool:
        return True


def make_moderator(devices=None, *, device_map=None, link_view=None, batch=None, sim_time: float = 0.0,
                   handler=None):
    """A running Moderator over `devices` (name -> device) without sockets or a stage. The device map lists the
    devices as articulations unless `device_map` says otherwise."""
    m = moderator_mod.Moderator(sim_context=FakeSimContext(current_time=sim_time), host="127.0.0.1",
                                subscribe_port=1, publish_port=2)
    m._message_handler = handler if handler is not None else FakeHandler()
    m._flag_running = True
    m._instance_map = dict(devices or {})
    if device_map is None:
        device_map = {moderator_mod.ARTICULATION: {name: {"path": getattr(d, "_path", f"/World/{name}")}
                                                   for name, d in m._instance_map.items()},
                      moderator_mod.RIGID_OBJECT: {}}
    m._device_map = device_map
    m._link_view, m._batch = link_view, batch
    return m
