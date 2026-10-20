"""Clean respawn: the PhysX root LINK goes back to its USD world pose, the joints to their authored state, and the
drive targets the re-placed pose.

The root link's pose is composed through every parent Xform and taken at the root body itself: the articulation
prim's local transform would put a robot under a group Xform at the origin, and a base body offset inside its prim
(a swerve base at z = 0.18) into the floor while the arms fixed to it stay put."""
from __future__ import annotations

import types

import numpy as np

from fakes import command, make_robot
from scripts import protocol
from scripts.device import articulation

W = articulation.WrappedArticulation


class _Meta:
    def __init__(self, names, parents):
        self.link_names = names
        self.link_parents = parents


class _PV:
    def __init__(self, links, parents=None, root_xf=None):
        self.link_paths = links
        if parents is not None:
            flat = links[0] if links and isinstance(links[0], list) else links
            self.shared_metatype = _Meta([p.rsplit("/", 1)[-1] for p in flat], parents)
        self._root_xf = root_xf

    def get_root_transforms(self):
        if self._root_xf is None:
            raise RuntimeError("no view")
        return np.asarray(self._root_xf, dtype=np.float32).reshape(1, 7)


def test_physx_root_link_is_the_parentless_link_of_the_view():
    pv = _PV([["/W/r/links/link_00", "/W/r/links/link_01"]], parents=["", "link_00"])
    assert W._physx_root_link_path(pv, "/W/r") == "/W/r/links/link_00"
    # PhysX order need not start at the root: the parentless link wins wherever it is
    pv = _PV([["/W/r/links/l1", "/W/r/links/body"]], parents=["body", None])
    assert W._physx_root_link_path(pv, "/W/r") == "/W/r/links/body"
    # a flat list and no metatype -> the first link
    pv = _PV(["/W/r/links/a", "/W/r/links/b"])
    assert W._physx_root_link_path(pv, "/W/r") == "/W/r/links/a"
    # a view that does not expose its links (or raises) -> None, the caller falls back

    class _NoLinks:
        @property
        def link_paths(self):
            raise RuntimeError("not available")

    assert W._physx_root_link_path(_NoLinks(), "/W/r") is None
    assert W._physx_root_link_path(_PV(None), "/W/r") is None
    assert W._physx_root_link_path(_PV([[]]), "/W/r") is None


def _respawn_setup(monkeypatch, world, links, parents, root_xf=None):
    a, pv_raw = make_robot(3)
    pv = _PV(links, parents, root_xf)
    for attr in ("get_dof_positions", "get_dof_velocities"):
        setattr(pv, attr, getattr(pv_raw, attr))
    a._art._physics_view = pv
    a._path = "/W/set2/r"
    monkeypatch.setattr(W, "_usd_world_pose", staticmethod(lambda stage, path: world.get(path)))
    calls = []
    a._art.set_world_poses = lambda positions, orientations: calls.append(("pose", np.array(positions), np.array(orientations)))
    a._art.set_joint_velocities = lambda v: calls.append(("dq", np.array(v)))
    a._art.set_joint_positions = lambda q: calls.append(("q", np.array(q)))
    a._art.set_velocities = lambda v: calls.append(("v", np.array(v)))
    return a, calls


def test_clean_respawn_places_the_root_link_at_its_usd_world_pose(monkeypatch):
    # a base body at z = 0.1815 inside a prim authored at the origin, under a group Xform at x + 4
    world = {
        "/W/set2/r/links/body": (np.array([[4.0, 0.0, 0.1815]], np.float32), np.array([1.0, 0.0, 0.0, 0.0], np.float32)),
        "/W/set2/r": (np.array([[4.0, 0.0, 0.0]], np.float32), np.array([1.0, 0.0, 0.0, 0.0], np.float32)),
    }
    a, calls = _respawn_setup(monkeypatch, world, [["/W/set2/r/links/body", "/W/set2/r/links/l1"]], ["", "body"],
                              root_xf=[4.0, 0.0, 0.1815, 0.0, 0.0, 0.0, 1.0])
    a._clean_respawn(a._art)
    assert [c[0] for c in calls] == ["pose", "dq", "q", "v"]
    np.testing.assert_allclose(calls[0][1], [[4.0, 0.0, 0.1815]])           # the root LINK's world pose ...
    np.testing.assert_allclose(calls[0][2], [[1.0, 0.0, 0.0, 0.0]])         # ... wxyz for Isaac Sim
    assert calls[0][1].shape == (1, 3) and calls[0][2].shape == (1, 4)
    np.testing.assert_allclose(calls[1][1], np.zeros((1, 3)))
    np.testing.assert_allclose(calls[2][1], np.zeros((1, 3)))               # no JointStateAPI: joints home at 0
    np.testing.assert_allclose(calls[3][1], np.zeros((1, 6)))


def test_clean_respawn_falls_back_to_the_prim_when_the_view_has_no_links(monkeypatch):
    world = {"/W/set2/r": (np.array([[4.0, 0.0, 0.0]], np.float32), np.array([1.0, 0.0, 0.0, 0.0], np.float32))}
    a, calls = _respawn_setup(monkeypatch, world, None, None)
    monkeypatch.setattr(W, "_first_rigid_body_path", staticmethod(lambda stage, path: None))
    a._clean_respawn(a._art)
    assert calls[0][0] == "pose"
    np.testing.assert_allclose(calls[0][1], [[4.0, 0.0, 0.0]])


def test_clean_respawn_leaves_the_base_alone_without_a_finite_spawn_pose(monkeypatch):
    a, calls = _respawn_setup(monkeypatch, {}, [["/W/set2/r/links/body"]], [""])
    a._clean_respawn(a._art)
    assert [c[0] for c in calls] == ["dq", "q", "v"]                       # joints and velocities still reset


def test_spawn_pose_reports_how_far_physx_had_drifted(monkeypatch):
    world = {"/W/set2/r/links/body": (np.array([[4.0, 0.0, 0.1815]], np.float32), np.array([1.0, 0.0, 0.0, 0.0], np.float32))}
    a, _ = _respawn_setup(monkeypatch, world, [["/W/set2/r/links/body"]], [""], root_xf=[4.3, 0.4, 0.1815, 0, 0, 0, 1])
    sp = a._spawn_root_pose(a._art, stage=None)
    assert sp.link == "/W/set2/r/links/body" and sp.source == "root link"
    assert abs(sp.moved - 0.5) < 1e-6


def test_clean_respawn_targets_the_re_placed_joints():
    # A timeline stop -> play re-places the robot while the command in force is still the one the gateway sent
    # before the stop, and the gateway sends again only after the first new state: the drive must target the
    # re-placed joints meanwhile.
    a, _ = make_robot(3)
    a._perm_arr = np.array([2, 0, 1])                 # DOF order differs from the actuator order
    a.update_target("sim_command", command(3, [0.9, -0.9, 0.5]))
    stale = a._q_d[:3].copy()
    a._last_drive = ("kp", "kd", "pos", "vel", "eff")
    a._hold_spawn_pose(np.array([0.11, 0.22, 0.33], dtype=np.float32))     # DOF order
    np.testing.assert_allclose(a._q_d[:3], a._to_canonical(np.array([0.11, 0.22, 0.33], dtype=np.float32)))
    assert not np.allclose(a._q_d[:3], stale)
    np.testing.assert_allclose(a._dq_d[:3], 0.0)
    np.testing.assert_allclose(a._tau_d[:3], 0.0)
    assert a._cmd_dirty and a._last_drive is None
    assert a._mode[:3].tolist() == [protocol.MODE_POSITION] * 3        # ownership and modes untouched


def test_joint_lookup_stays_inside_the_robot(monkeypatch):
    """The joints of /World/go1 are those under /World/go1, not every joint whose path starts with it: /World/go1_2
    (same joint names) must not shadow them."""
    import pxr

    class _Joint:
        def __init__(self, path):
            self.path = path

        def GetPath(self):
            return self.path

        def GetName(self):
            return self.path.rsplit("/", 1)[-1]

        def GetTypeName(self):
            return "PhysicsRevoluteJoint"

    subtree = {"/World/go1": [_Joint("/World/go1/joints/hip"), _Joint("/World/go1/joints/knee")],
               "/World/go1_2": [_Joint("/World/go1_2/joints/hip"), _Joint("/World/go1_2/joints/knee")]}
    stage = types.SimpleNamespace(GetPrimAtPath=lambda path: path)       # a prim stands for its path here
    monkeypatch.setattr(articulation.omni.usd, "get_context", lambda: types.SimpleNamespace(get_stage=lambda: stage))
    monkeypatch.setattr(pxr.Usd, "PrimRange", lambda root: iter(subtree[root]))
    a, _ = make_robot(2, path="/World/go1")
    joints = a._joints_by_name()
    assert {name: j.GetPath() for name, j in joints.items()} == {"hip": "/World/go1/joints/hip",
                                                                 "knee": "/World/go1/joints/knee"}
