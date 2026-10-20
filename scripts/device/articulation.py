"""One robot: an Isaac Sim articulation driven like a set of servo drives by the controller's actuator gateways.

Each articulation DOF is one actuator. A gateway sends actuator-space commands (SimCommand) and reads back
actuator-space state (SimState), exactly as from a servo drive with its motor and encoder; the kinematic
transmission lives in the controller above the gateway.

Per DOF: the four drive modes (POSITION, VELOCITY, TORQUE, IMPEDANCE); MODE_IDLE (0xFE), so several gateways share
one articulation through disjoint DOF windows and uncommanded casters roll freely; MODE_INIT (0xFF) teleports.
Actuator i is the i-th movable USD joint under the robot prim in authoring order (rc:passive joints left out),
mapped onto PhysX's DOF order, which follows the articulation topology, at every Isaac Sim API boundary.

Optional robot attributes (on the articulation root unless noted):

    rc:positionKp, rc:positionKd                    POSITION-mode drive gains, one value per actuator
    rc:positionVelocityFeedforward                  with authored gains: feed the target velocity to the drive
    rc:positionDriveModel = "capped_p_passive"      explicit position torque, P saturated before the losses,
      rc:positionTorqueLimits, rc:positionFrictionLoss   one value per actuator
    rc:imuLink, rc:imuOffset, rc:imuAccelTau        emulated IMU -> imu_state
    rc:contactLinks, rc:contactOffsets, rc:contactHeight, rc:groundZ   foot-contact flags in imu_state
    rc:contactForceLinks                            measured contact loads -> contact_forces
    rc:actuator:defaults, and on joints rc:actuator:model / :params / :name   servo models (actuator_model.py)
    on joints: rc:passive, rc:backlashOf            joints outside the actuator set (backlash hinges, casters)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
import omni.usd
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.types import ArticulationActions

from .. import config, protocol
from ..config import IDLE_KD, POSITION_KD, POSITION_KP, TORQUE_KD, VELOCITY_KD
from ..imu_batch import GRAVITY, quat_to_matrix_wxyz
from .actuator_model import ATTR_MODEL, MODEL_BAM, ActuatorModelSet, read_actuator_specs
from .device import WrappedDevice
from .root_pose import ROOT_POSE_TOPIC, RootPoseRate, encode_root_pose, root_pose_hz_from_env

if TYPE_CHECKING:
    from typing import Dict, List, Tuple
    from numpy import ndarray
    from isaacsim.core.api import SimulationContext
    from pxr.Usd import Prim

# A drive force limit at or above this is PhysX's "unlimited" (FLT_MAX for an unauthored maxForce), not a limit.
_EFFORT_LIMIT_SENTINEL = 1.0e30
# Physics steps between two checks that this robot's row of the stage-wide articulation batch still reports its own
# joint state (see _verify_batch_row): one extra per-robot read per second at 250 Hz.
_BATCH_RECHECK_STEPS = 250
_IDX0 = np.zeros(1, dtype=np.uint32)     # "articulation 0 of the view" index for the raw view's set_* calls
_MOVABLE_JOINTS = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")


def _attr_value(prim, name: str):
    """The value of attribute `name` on `prim`, None when there is no such attribute."""
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.IsValid() else None


class SpawnPose(NamedTuple):
    """Where a clean respawn puts the PhysX root link (see WrappedArticulation._spawn_root_pose)."""
    pos: "ndarray"        # (1, 3) float32 world position
    quat: "ndarray"       # (4,) float32 world orientation, wxyz
    link: str             # prim path the pose was read from
    source: str           # "root link" | "first rigid body" | "articulation prim"
    moved: "Optional[float]"   # metres between that pose and the root PhysX reported, None if unknown


class WrappedArticulation(WrappedDevice):
    """One articulation of the stage, bound to its PhysX view (see the module docstring)."""

    # BAM servo models were validated at 200 Hz physics; at 100 Hz the servo loop chatters at the Nyquist rate.
    AM_MAX_VALIDATED_DT = 0.005

    def __new__(cls, sim_context: "SimulationContext", path: str, name: str):
        instance = super().__new__(cls)
        device = Articulation(path, name=name)
        if not device:
            return None
        # Articulation() binds its PhysX view only when a simulation view exists at construction time; before the
        # first physics warm-up it comes back with num_dof None. Refuse it so the moderator retries instead of
        # keeping a handle that can never read or drive.
        if device.num_dof is None or not device.is_physics_handle_valid():
            return None
        instance._sim_context = sim_context
        instance._name = name
        instance._path = path
        instance._art = device
        instance._n = device.num_dof
        return instance

    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger("isaac_bridge")
        self._latch_enabled = config.frame_latch_enabled()
        self._respawn_at_bind = config.respawn_at_bind()

        # Merged command state: SIM_MAX_JOINTS canonical slots, allocated on the first command; MODE_IDLE = no
        # gateway owns that DOF. Several gateways may share this articulation, each driving its own DOF window
        # (sim.transport.dof_offset, e.g. arm [0..3] + gripper [4..5]): commands are merged per DOF, so their
        # streams coexist instead of overwriting each other.
        self._mode = None         # (SIM_MAX_JOINTS,) uint8, MODE_IDLE where unowned
        self._q_d = None          # (SIM_MAX_JOINTS,) float32, like dq_d, tau_d, kp, kd
        self._dq_d = None
        self._tau_d = None
        self._kp = None
        self._kd = None
        self._k = None            # largest command window (num_joints) seen across gateways
        self._tp_mask = None      # (SIM_MAX_JOINTS,) bool: DOFs with a pending 0xFF teleport
        self._tp_q = None         # (SIM_MAX_JOINTS,) float32 teleport pose of the masked DOFs
        self._nonfinite_cmds = 0
        self._latched = None      # frame latch: (mode, q_d, dq_d, tau_d, kp, kd) in force for the current frame
        # Drive targets are re-sent to PhysX only when something changed them (a command, a teleport, a physics
        # re-init, the frame latch) — and every step while a servo model computes efforts or a DOF asks for a
        # feed-forward effort. Position / velocity targets and gains persist in PhysX between steps.
        self._cmd_dirty = True
        self._last_kp = None      # the gains PhysX holds, written only when they change
        self._last_kd = None
        self._last_drive = None   # (kp, kd, pos_t, vel_t, eff) of the last drive, canonical order
        self._drive_has_effort = False

        # State and publishing
        self._actual = None       # (q, dq, tau) float64, canonical order
        self._seq = 0
        self._state_packer = protocol.StatePacker()
        self._imu_packer = protocol.ImuPacker()
        self._fmax_canon = None   # drive force limit per DOF (canonical), False when none is usable
        self._effort_warned = False

        # Physics re-initialization: a timeline stop -> play on the same stage keeps this device while PhysX
        # re-parses the USD, so everything written through the tensor API (drive gains, armature, velocity clamp)
        # silently reverts to its USD value while the caches above still claim it is in place. synchronize() watches
        # the physics view identity (and the moderator's on_resume()) and re-applies it (_on_physics_reinit).
        self._phys_view = None    # strong reference: a freed view's address could be reused by the new one
        self._reinit_pending = False
        self._respawn_left = 2 if self._respawn_at_bind else 0   # clean respawn two physics steps after a (re)bind

        # Tensor views
        self._raw_ok = None       # None = not probed yet; True/False = the raw tensor view is numpy-backed
        self._link_view = None    # stage-wide rigid-body view (link_view.LinkStateView)
        self._imu_rows = None     # (IMU row, [contact rows]) in self._link_view for generation self._imu_rows_gen
        self._imu_rows_gen = -1
        self._batch = None        # stage-wide articulation view (articulation_batch.ArticulationBatch)
        self._batch_row = -1
        self._batch_gen = -1
        self._batch_check_left = 0
        self._batch_verified = False
        self._batch_step_seen = object()

        # root_pose frame (device/root_pose.py): the PhysX root link's world pose, read only on the steps that
        # publish it — from its row in the link view, else from the articulation view's root transform
        self._root_rate = RootPoseRate(root_pose_hz_from_env())
        self._root_seq = 0
        self._root_link = None    # (prim path, source) of the PhysX root link once resolved
        self._root_row = -1       # its row in self._link_view for generation self._root_row_gen
        self._root_row_gen = -1
        self._root_source = None  # "link view" | "articulation view", logged when it changes
        self._root_skips = 0      # non-finite poses not published

        # Actuator <-> PhysX DOF mapping, built from the USD joint order once the physics view is live
        self._perm_arr = None     # perm[i] = PhysX DOF index of actuator i
        self._n_dof = None        # PhysX DOF count (more than the actuators when the USD has rc:passive joints)
        self._n_dof_perm = None   # the DOF count the mapping was built for
        self._passive_kd = None   # (n_dof,) USD drive damping of the passive DOFs [Nm s/rad], 0 elsewhere
        self._bl_dof = None       # (n,) PhysX DOF of the backlash hinge behind actuator i, -1 = none

        # Robot metadata: the rc:* attributes, read once the physics view is live
        self._meta_loaded = False
        self._usd_kp = None       # POSITION gains from rc:positionKp / rc:positionKd (canonical); None = defaults
        self._usd_kd = None
        self._usd_vel_ff = False
        self._capped_position = None   # capped-P position drive: {"limits", "friction"} per actuator
        self._capped_tau = None        # (mask, torque) the capped drive applied this step, canonical order
        self._contact_force_paths = None   # rc:contactForceLinks prim paths (read by the moderator's pass)
        self._contact_force_payload = None
        # IMU and foot-contact emulation: None until the metadata is read, False when the robot has none
        self._imu = None
        self._imu_seq = 0
        self._imu_payload = None
        self._imu_batched = False
        # Servo actuator models (device/actuator_model.py): POSITION-mode DOFs driven by the model's joint efforts
        # instead of the PhysX PD drive; IDLE = unpowered servo (friction only)
        self._am = None
        self._am_tau = None
        self._am_joints = None    # joint prims (canonical order) the models were read from
        self._am_ext = True       # PhysX gravity / Coriolis as the external-torque estimate (off after a failure)
        self._am_inertia = None   # (n,) apparent joint inertia from the PhysX mass matrix (canonical)
        self._am_inertia_ok = True
        self._am_step = 0
        self._am_logged = False

        self._logged_apply = False   # one DEBUG readback of the first drive (_log_first_drive)

    # -- WrappedDevice ------------------------------------------------------------
    def destroy(self) -> None:
        try:
            self._art.__del__()      # Isaac's prim wrappers deregister their callbacks and drop their view here
        except Exception:
            pass

    @classmethod
    def is_articulation(cls, prim: "Prim") -> bool:
        return bool(prim.HasAttribute("physxArticulation:articulationEnabled"))

    def update_target(self, data_type: str, data: bytes) -> None:
        if data_type == "sim_command":
            self._handle_sim_command(data)

    def get_actual(self) -> "List[Tuple[str, bytes]]":
        if self._actual is None:
            return []
        q, dq, tau = self._actual
        self._seq += 1
        # Every live DOF, not just the commanded window: each gateway slices its own window, and a dof_offset
        # gateway sends nothing until it sees its window in the state, so publishing only the commanded DOFs would
        # leave it waiting for ever.
        k = min(len(q), protocol.SIM_MAX_JOINTS)
        try:
            sim_time = float(self._sim_context.current_time)
        except (TypeError, ValueError):
            sim_time = 0.0
        frames = [("sim_state", self._state_packer.pack(int(sim_time * 1e9), self._seq, k, q[:k], dq[:k], tau[:k]))]
        if self._imu_payload is not None:
            frames.append((protocol.IMU_STATE_TOPIC, self._imu_payload))
        if self._contact_force_payload is not None:
            frames.append((protocol.CONTACT_FORCES_TOPIC, self._contact_force_payload))
        # root_pose goes last and never costs the frames above: a failure here only skips this frame
        try:
            root = self._root_pose_frame(sim_time)
        except Exception as e:
            root = None
            self._logger.debug(f"[{self._name}] root_pose skipped: {type(e).__name__}: {e}")
        if root is not None:
            frames.append((ROOT_POSE_TOPIC, root))
        return frames

    # -- moderator hooks ------------------------------------------------------------
    def set_link_view(self, view) -> None:
        """The stage-wide rigid-body view (None = per-prim reads)."""
        self._link_view = view
        self._imu_rows = None
        self._imu_rows_gen = -1

    def set_batch(self, batch) -> None:
        """The stage-wide articulation batch (None = this robot's own tensor view)."""
        self._batch = batch
        self._batch_row = -1
        self._batch_gen = -1
        self._batch_check_left = 0
        self._batch_verified = False
        self._batch_step_seen = object()

    def on_resume(self) -> None:
        """The timeline was started again with this device kept. The re-application waits for the next physics
        step, when the new view is live; a device that never synchronized has nothing to restore."""
        if self._phys_view is not None:
            self._reinit_pending = True
        self._cmd_dirty = True

    def latch_command(self) -> None:
        """Frame latch: snapshot the merged command for the frame that starts now (first physics step)."""
        if not self._latch_enabled or self._q_d is None:
            return
        self._latched = (self._mode.copy(), self._q_d.copy(), self._dq_d.copy(), self._tau_d.copy(),
                         self._kp.copy(), self._kd.copy())
        self._cmd_dirty = True

    def has_imu(self) -> bool:
        """True once the robot's rc:imuLink has been read and it emulates an IMU."""
        return bool(self._imu)

    def imu_binding(self):
        """What the stage-wide ImuBatch needs to compute this robot's IMU, or None (no rc:imuLink, not ready)."""
        imu = self._imu
        if not imu or not imu.get("path"):
            return None
        return {
            "path": imu["path"],
            "contact_paths": list(imu.get("contact_paths") or []),
            "offset": np.asarray(imu.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64),
            "com_local": np.asarray(imu.get("com_local", (0.0, 0.0, 0.0)), dtype=np.float64),
            "contact_offsets": [np.asarray(off, dtype=np.float64) for _, off in imu.get("contacts", [])],
            "contact_height": float(imu.get("contact_height", 0.01)),
            "ground_z": float(imu.get("ground_z", 0.0)),
            "accel_tau": float(imu.get("accel_tau", 0.02)),
        }

    def set_imu_batched(self, on: bool) -> None:
        """True = the stage-wide ImuBatch computes this robot's IMU; the per-robot pass is skipped."""
        self._imu_batched = bool(on)

    def set_imu_blob(self, blob: bytes, seq: int) -> None:
        """This robot's imu_state record, packed by the stage-wide pass."""
        self._imu_seq = int(seq)
        self._imu_payload = blob

    def contact_force_binding(self):
        """Link prim paths for the stage-wide ContactForceBatch, or None (no rc:contactForceLinks)."""
        paths = self._contact_force_paths
        return list(paths) if paths else None

    def set_contact_force_payload(self, blob: bytes) -> None:
        self._contact_force_payload = blob

    # -- commands -------------------------------------------------------------------
    def _ensure_buffers(self) -> None:
        if self._mode is not None:
            return
        M = protocol.SIM_MAX_JOINTS
        self._mode = np.full(M, protocol.MODE_IDLE, dtype=np.uint8)
        self._q_d = np.zeros(M, dtype=np.float32)
        self._dq_d = np.zeros(M, dtype=np.float32)
        self._tau_d = np.zeros(M, dtype=np.float32)
        self._kp = np.zeros(M, dtype=np.float32)
        self._kd = np.zeros(M, dtype=np.float32)
        self._tp_mask = np.zeros(M, dtype=bool)
        self._tp_q = np.zeros(M, dtype=np.float32)

    def _handle_sim_command(self, data: bytes) -> None:
        try:
            cmd = protocol.parse_command(data)
        except ValueError as e:
            self._logger.warning(f"[{self._name}] bad SimCommand: {e}")
            return
        M = protocol.SIM_MAX_JOINTS
        self._ensure_buffers()

        # A gateway commands the window [0, num_joints) minus its MODE_IDLE slots (a dof_offset gateway marks the
        # slots below its window IDLE). Entries this command does not own are left untouched; DOFs no gateway ever
        # owns (casters, idle wheels) stay MODE_IDLE and roll freely.
        kc = max(0, min(int(cmd.num_joints), M))
        self._note_command_window(kc)
        # Never let a non-finite value reach PhysX: a gateway that read a NaN state echoes it back as its hold
        # target, and a single NaN target turns the whole articulation NaN for good. DOFs carrying a non-finite
        # q_d / dq_d / tau_d / kp / kd keep their previous command.
        finite = self._finite_dofs(cmd, kc)
        if finite is not None and not finite.any():
            return
        # Fast path: every DOF of the window in a drive mode (no teleport, no IDLE hole), which is what a gateway
        # commanding its whole window sends every cycle — contiguous slice copies instead of masked scatters.
        mode_w = cmd.mode[:kc]
        if kc and finite is None and not ((mode_w == protocol.MODE_INIT) | (mode_w == protocol.MODE_IDLE)).any():
            self._mode[:kc] = mode_w
            self._q_d[:kc] = cmd.q_d[:kc]
            self._dq_d[:kc] = cmd.dq_d[:kc]
            self._tau_d[:kc] = cmd.tau_d[:kc]
            self._kp[:kc] = cmd.kp[:kc]
            self._kd[:kc] = cmd.kd[:kc]
            self._cmd_dirty = True
            return

        window = np.zeros(M, dtype=bool)
        window[:kc] = True
        if finite is not None:
            window[:kc] &= finite                     # dropped DOFs: neither teleported nor driven
        init_m = window & (cmd.mode == protocol.MODE_INIT)
        drive_m = window & (cmd.mode != protocol.MODE_INIT) & (cmd.mode != protocol.MODE_IDLE)
        if init_m.any():
            # set_initial_state(): teleport those DOFs to q_d (applied in synchronize)
            self._tp_mask |= init_m
            self._tp_q[init_m] = cmd.q_d[init_m].astype(np.float32)
        if drive_m.any():
            self._mode[drive_m] = cmd.mode[drive_m]
            self._q_d[drive_m] = cmd.q_d[drive_m].astype(np.float32)
            self._dq_d[drive_m] = cmd.dq_d[drive_m].astype(np.float32)
            self._tau_d[drive_m] = cmd.tau_d[drive_m].astype(np.float32)
            self._kp[drive_m] = cmd.kp[drive_m].astype(np.float32)
            self._kd[drive_m] = cmd.kd[drive_m].astype(np.float32)
        if init_m.any() or drive_m.any():
            self._cmd_dirty = True

    def _note_command_window(self, kc: int) -> None:
        if self._k is None or kc > self._k:
            self._logger.info(f"[{self._name}] gateways command {kc} DOF; DOFs no gateway owns roll freely "
                              f"(light damping)")
            self._k = kc

    def _finite_dofs(self, cmd, kc: int):
        """None when every value of the command window is finite (the normal case, one check per array); else the
        per-DOF mask of the DOFs whose q_d, dq_d, tau_d, kp and kd are all finite. Logs the first drop and every
        1000th after it, naming the offending fields."""
        if not kc:
            return None
        arrays = ((cmd.q_d, "q_d"), (cmd.dq_d, "dq_d"), (cmd.tau_d, "tau_d"), (cmd.kp, "kp"), (cmd.kd, "kd"))
        ok = None
        bad_fields = []
        for arr, name in arrays:
            f = np.isfinite(arr[:kc])
            if f.all():
                continue
            bad_fields.append(name)
            ok = f if ok is None else (ok & f)
        if ok is None:
            return None
        self._nonfinite_cmds += 1
        if self._nonfinite_cmds == 1 or self._nonfinite_cmds % 1000 == 0:
            self._logger.warning(f"[{self._name}] command with non-finite {'/'.join(bad_fields)} for DOF(s) "
                                 f"{np.flatnonzero(~ok).tolist()} dropped ({self._nonfinite_cmds} so far): the "
                                 f"controller probably read a non-finite state; those DOFs keep their previous command")
        return ok

    # -- per physics step -----------------------------------------------------------
    def synchronize(self) -> None:
        art = self._art
        batch = self._batch_view(art)
        # the robot's own view is only probed when the batch does not carry it; the rare paths that still need it
        # (teleport, the servo model's gravity / mass matrix) ask for it themselves
        pv = None if batch is not None else self._raw_view(art)

        # Joint positions first: the live read also gives the DOF count (num_dof can be None at bind time).
        try:
            if batch is not None:
                q_dof = batch.positions(self._batch_row, self._dof_count(art))    # view of the shared buffer
            elif pv is not None:
                q_dof = np.array(pv.get_dof_positions()[0], dtype=np.float64)    # copy: the view reuses its buffer
            else:
                q_dof = np.asarray(art.get_joint_positions()[0], dtype=np.float64)
        except Exception as e:
            self._logger.debug(f"[{self._name}] not ready yet: {e}")
            return
        n_dof = len(q_dof)
        self._n_dof = n_dof
        # PhysX orders DOFs by articulation topology, not by the USD joint order the controller's actuator indices
        # follow: from here on every array crossing the Isaac Sim API is gathered DOF -> actuator on read and
        # scattered actuator -> DOF on write.
        self._ensure_perm(art, n_dof)
        n = self._n
        self._ensure_robot_meta(n)
        self._check_physics_reinit(art)
        if self._respawn_left > 0:
            self._respawn_left -= 1
            if self._respawn_left == 0:
                self._clean_respawn(art)        # the robot is placed; this step's drive targets it at home
        batch = self._batch_view(art)           # a physics re-init swaps the views behind the batch
        pv = None if batch is not None else self._raw_view(art)

        # actuator order from here on: q_enc = what the servo encoder (and the policy) sees, through the backlash
        # hinge when the USD has one; q = the motor-side angle the PhysX drive acts on
        q = self._to_canonical(q_dof)
        q_enc = self._through_backlash(q, q_dof)
        dq_dof = self._read_dq(art, pv, batch)  # read once: the servo model, the effort and the publish share it
        dq_motor = self._to_canonical(dq_dof)

        self._capped_tau = None
        if self._tp_mask is not None and bool(self._tp_mask[:n].any()):
            # a teleport (initial pose, hot switch) wins over driving
            self._apply_teleport()
            # the teleport zeroed the velocities: the batch snapshot is stale, read this robot's own view
            dq_dof = self._read_dq(art, pv if pv is not None else self._raw_view(art), None)
            dq_motor = self._to_canonical(dq_dof)
        elif self._mode is not None and (bool(np.any(self._mode[:n] != protocol.MODE_IDLE)) or self._drive_has_effort):
            self._drive(art, pv, batch, n_dof, q, q_enc, q_dof, dq_dof, dq_motor)

        dq = self._through_backlash(dq_motor, dq_dof)
        self._actual = (q_enc, dq, self._published_effort(art, batch, n, q_enc, dq_motor))
        # IMU and foot contacts (robots that author rc:imuLink); the moderator's ImuBatch computes them for every
        # robot at once when it carries this one, so this is only the fallback
        if self._imu and not self._imu_batched:
            self._sample_imu()

    def _drive(self, art, pv, batch, n_dof: int, q, q_enc, q_dof, dq_dof, dq_motor) -> None:
        """Drive from the merged command: resolve the PhysX drive, apply the servo / capped models, and write gains
        and targets — only when something changed or a model computes per-step efforts."""
        n = self._n
        if not (self._cmd_dirty or self._am is not None or self._capped_position is not None
                or self._drive_has_effort or self._last_drive is None):
            return
        kp, kd, pos_t, vel_t, eff = self._resolve_drive(q)
        self._last_drive = (kp, kd, pos_t, vel_t, eff)    # the models below adjust these arrays in place
        if self._am is not None:
            # gravity / Coriolis / mass matrix are per-robot reads: they need this robot's own view
            self._apply_actuator_models(art, n, q_enc, kp, kd, vel_t, pos_t, eff,
                                        pv=(pv if pv is not None else self._raw_view(art)), dq_dof=dq_dof)
        if self._capped_position is not None:
            self._apply_capped_position(q, dq_motor, kp, kd, vel_t, pos_t, eff)
        if self._gains_changed(kp, kd):
            # (1, n) arrays, not (n,): set_gains documents (M, K) and its USD fallback indexes kps[m][k]
            try:
                self._write_gains(art, pv, batch, self._to_dof(kp).reshape(1, n_dof),
                                  self._to_dof(kd, base=self._passive_kd).reshape(1, n_dof))
                self._last_kp, self._last_kd = kp.copy(), kd.copy()
            except Exception as e:
                # a WARNING: a lost gain write leaves every joint at whatever stiffness the USD authored, which
                # looks exactly like "the robot ignores commands"
                self._logger.warning(f"[{self._name}] set_gains failed: {type(e).__name__}: {e}")
                self._last_kp = self._last_kd = None
        # Efforts only when some DOF asks for one (or a servo model drives): an actuation force written every step,
        # even zero, competes with the implicit PD drive and the joint sags under gravity.
        eff_dof = self._to_dof(eff)
        send_eff = bool(np.any(eff_dof)) or self._am is not None
        # a DOF that stops asking for a feed-forward torque must have it cleared: PhysX keeps applying an actuation
        # force until it is rewritten
        clear_eff = self._drive_has_effort and not send_eff
        self._drive_has_effort = send_eff
        pos_dof = self._to_dof(pos_t, base=q_dof.astype(np.float32))
        vel_dof = self._to_dof(vel_t)
        self._write_targets(art, pv, batch, pos_dof.reshape(1, n_dof), vel_dof.reshape(1, n_dof),
                            eff_dof.reshape(1, n_dof) if send_eff else None, clear_eff=clear_eff)
        self._cmd_dirty = False
        self._log_first_drive(art, kp, kd, pos_t)

    def _published_effort(self, art, batch, n: int, q, dq_motor) -> np.ndarray:
        """The joint efforts published in sim_state (canonical order).

        A PD-driven DOF publishes its drive torque — kp (target - q) + kd (vel_t - dq) + feed-forward, clamped to
        the drive's maxForce — which is what a motor driver reports. The measured joint effort is the joint's
        reaction torque instead, with every foot-impact spike in it (several times an actuator's rating), and a
        controller's torque supervision trips on those. Only a DOF whose drive has a force limit publishes the
        estimate: without one, an authored stiffness turns a sub-degree hold error into kilonewton-metres, so those
        DOFs (and DOFs with kp = 0) publish the measured reaction torque. Servo-model and capped-P DOFs publish the
        torque their model applied."""
        measured_read = self._measured_effort_needed(n)
        if measured_read:
            try:
                tau = self._to_canonical(self._read_measured_efforts(art, batch))
            except Exception:
                tau = np.zeros(n, dtype=np.float64)
                measured_read = False
        else:
            tau = np.zeros(n, dtype=np.float64)      # every DOF is overwritten below: skip the tensor read
        if self._last_drive is not None:
            try:
                tau = self._estimate_drive_effort(n, q, dq_motor, self._max_efforts(art), tau)
            except Exception as e:
                tau = self._effort_fallback(art, n, tau, measured_read, e)
        if self._am is not None and self._am_tau is not None:
            tau = np.where(self._am.mask, self._am_tau, tau)
        capped = self._capped_tau
        if capped is not None and self._capped_position is not None:
            tau = np.where(capped[0][:n], capped[1][:n], tau)
        return tau

    # -- drive resolution -------------------------------------------------------------
    def _command_in_force(self):
        """(mode, q_d, dq_d, tau_d, kp, kd) the drive follows this step: the frame's snapshot when the frame latch
        is on, else the live merge buffers."""
        if self._latch_enabled and self._latched is not None:
            return self._latched
        return (self._mode, self._q_d, self._dq_d, self._tau_d, self._kp, self._kd)

    def _resolve_drive(self, q_now: np.ndarray):
        """Per-DOF SimCommand modes -> one PhysX drive (kp, kd, pos_t, vel_t, eff), canonical order, owned float32
        arrays (the servo models adjust them in place).

        PhysX joint drive: effort = kp (pos_t - q) + kd (vel_t - dq) + applied effort, so one gains + targets write
        covers all four modes by choosing per-DOF gains and targets:

            POSITION   stiff kp / kd, pos_t = q_d       (servo tracking)
            VELOCITY   kp = 0, kd = VELOCITY_KD, vel_t = dq_d
            TORQUE     kp = 0, kd = TORQUE_KD, effort = tau_d
            IMPEDANCE  kp / kd from the command, pos_t = q_d, vel_t = dq_d, effort = tau_d
        """
        n = self._n
        # the command arrays are sliced to the live DOF count: a command stored before num_dof was known is
        # SIM_MAX_JOINTS long
        mode, q_d, dq_d, tau_d, kp_c, kd_c = (a[:n] for a in self._command_in_force())
        if n and np.all(mode == protocol.MODE_POSITION):
            return self._resolve_position(n, q_d, dq_d, tau_d)
        return self._resolve_mixed(n, q_now, mode, q_d, dq_d, tau_d, kp_c, kd_c)

    def _resolve_position(self, n: int, q_d, dq_d, tau_d):
        """Every DOF in POSITION mode (a walking or arm gateway): _resolve_mixed without the masked gathers."""
        authored = self._usd_kp is not None and len(self._usd_kp) == n
        kp = np.array(self._usd_kp, dtype=np.float32, copy=True) if authored else np.full(n, POSITION_KP, dtype=np.float32)
        kd = np.array(self._usd_kd, dtype=np.float32, copy=True) if authored else np.full(n, POSITION_KD, dtype=np.float32)
        pos_t = np.array(q_d, dtype=np.float32, copy=True)
        if authored and not self._usd_vel_ff:
            vel_t, eff = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        else:
            vel_t = np.array(dq_d, dtype=np.float32, copy=True)
            eff = np.array(tau_d, dtype=np.float32, copy=True)
        return kp, kd, pos_t, vel_t, eff

    def _resolve_mixed(self, n: int, q_now, mode, q_d, dq_d, tau_d, kp_c, kd_c):
        q_now = q_now.astype(np.float32)
        kp = np.zeros(n, dtype=np.float32)
        kd = np.zeros(n, dtype=np.float32)
        pos_t = q_now.copy()                 # velocity / torque DOFs: hold here
        vel_t = np.zeros(n, dtype=np.float32)
        eff = np.zeros(n, dtype=np.float32)

        is_pos = mode == protocol.MODE_POSITION
        is_vel = mode == protocol.MODE_VELOCITY
        is_tau = mode == protocol.MODE_TORQUE
        is_imp = mode == protocol.MODE_IMPEDANCE

        kp[is_pos] = POSITION_KP
        kd[is_pos] = POSITION_KD
        pos_t[is_pos] = q_d[is_pos]
        vel_t[is_pos] = dq_d[is_pos]
        eff[is_pos] = tau_d[is_pos]
        if self._usd_kp is not None and len(self._usd_kp) == n:
            # Robot-authored servo gains model a position servo whose damping is passive (-kd dq), as in the MJCF
            # a policy is trained on. Feeding the controller's interpolated target velocity into the drive
            # (kd (dq_d - dq)) injects a torque impulse every command period; rc:positionVelocityFeedforward = true
            # turns it on again.
            kp[is_pos] = self._usd_kp[is_pos]
            kd[is_pos] = self._usd_kd[is_pos]
            if not self._usd_vel_ff:
                vel_t[is_pos] = 0.0
                eff[is_pos] = 0.0

        kd[is_vel] = VELOCITY_KD
        vel_t[is_vel] = dq_d[is_vel]
        eff[is_vel] = tau_d[is_vel]

        kd[is_tau] = TORQUE_KD
        eff[is_tau] = tau_d[is_tau]

        kp[is_imp] = kp_c[is_imp]
        kd[is_imp] = kd_c[is_imp]
        pos_t[is_imp] = q_d[is_imp]
        vel_t[is_imp] = dq_d[is_imp]
        eff[is_imp] = tau_d[is_imp]

        # DOFs no gateway drives (passive casters, idle wheels, another gateway's window before its first
        # command): a zeroed entry would read as MODE_POSITION (mode 0) and lock them at 0. They roll freely
        # instead: kp = 0 and a light damping (IDLE_KD) like bearing friction.
        is_idle = mode == protocol.MODE_IDLE
        if is_idle.any():
            kp[is_idle] = 0.0
            kd[is_idle] = IDLE_KD
            pos_t[is_idle] = q_now[is_idle]
            vel_t[is_idle] = 0.0
            eff[is_idle] = 0.0
        return kp, kd, pos_t, vel_t, eff

    def _gains_changed(self, kp: np.ndarray, kd: np.ndarray) -> bool:
        if self._last_kp is None or self._last_kd is None:
            return True
        return not (np.array_equal(self._last_kp, kp) and np.array_equal(self._last_kd, kd))

    def _apply_capped_position(self, q, dq, kp, kd, vel_t, pos_t, eff) -> None:
        """Explicit motor torque for POSITION commands, evaluated every physics step (rc:positionDriveModel =
        "capped_p_passive"): the P torque saturates before damping and friction are deducted, and a second cap keeps
        the total within the torque limit even while braking. Other modes keep their drive."""
        n = self._n
        mask = self._command_in_force()[0][:n] == protocol.MODE_POSITION
        if not mask.any():
            return
        spec = self._capped_position
        limits = spec["limits"][mask]
        torque = np.clip(kp[mask] * (pos_t[mask] - q[mask]), -limits, limits)
        torque -= kd[mask] * dq[mask]
        torque -= spec["friction"][mask] * np.tanh(dq[mask] / .02)
        eff[mask] = np.clip(torque, -limits, limits)
        kp[mask] = kd[mask] = vel_t[mask] = 0.0
        # the published effort of these DOFs is the motor torque applied (with kp zeroed they would otherwise
        # publish the joint reaction torque, see _published_effort)
        self._capped_tau = (mask.copy(), np.asarray(eff[:n], dtype=np.float64).copy())

    def _apply_actuator_models(self, art, n: int, q, kp, kd, vel_t, pos_t, eff, pv=None, dq_dof=None) -> None:
        """Replace the PhysX drive by the servo model on the modelled DOFs, in place on the resolved drive arrays:
        POSITION = powered servo tracking pos_t, IDLE = unpowered (friction only); other modes keep the drive. The
        efforts computed here are applied as joint actuation forces."""
        am = self._am
        mode = self._command_in_force()[0][:n]
        m = am.mask & ((mode == protocol.MODE_POSITION) | (mode == protocol.MODE_IDLE))
        if not m.any():
            return
        try:
            if dq_dof is None:
                dq_dof = self._read_dq(art, pv)
            dq = self._to_canonical(np.asarray(dq_dof, dtype=np.float64))
        except Exception:
            return
        batch = self._batch if (self._batch is not None and self._batch_row >= 0 and self._batch.ok) else None
        src = pv if pv is not None else art
        ext = None
        if self._am_ext:
            try:
                n_dof = self._n_dof or n
                g = c = None
                if batch is not None:
                    gb = batch.gravity_forces(self._batch_row, n_dof)
                    cb = batch.coriolis_forces(self._batch_row, n_dof)
                    if gb is not None and cb is not None:
                        g = np.asarray(gb, dtype=np.float64).reshape(-1)
                        c = np.asarray(cb, dtype=np.float64).reshape(-1)
                if g is None or c is None:
                    g = np.asarray(src.get_generalized_gravity_forces()[0], dtype=np.float64).reshape(-1)
                    c = np.asarray(src.get_coriolis_and_centrifugal_forces()[0], dtype=np.float64).reshape(-1)
                if len(g) >= n_dof and len(c) >= n_dof:
                    # BAM sign convention: the external torque is -(gravity + Coriolis compensation)
                    ext = -self._to_canonical(g[-n_dof:] + c[-n_dof:])
            except Exception as e:
                self._am_ext = False
                self._logger.info(f"[{self._name}] actuator model: no gravity/Coriolis estimate ({e})")
        # Apparent joint inertia (mass-matrix diagonal) for the MuJoCo-style stopping-torque friction; without it the
        # model uses its dissipative band form. Refreshed every 10 physics steps. A floating base adds 6 root DOFs
        # in front, whose three translational entries all equal the robot mass.
        self._am_step += 1
        if self._am_inertia_ok and am.wants_inertia() and (self._am_inertia is None or self._am_step % 10 == 0):
            try:
                n_dof = self._n_dof or n
                M = None
                if batch is not None:
                    mb = batch.mass_matrix(self._batch_row, n_dof)
                    if mb is not None:
                        M = np.asarray(mb, dtype=np.float64)
                if M is None:
                    M = np.asarray(src.get_mass_matrices()[0], dtype=np.float64)
                diag = np.diagonal(M.reshape(M.shape[-2], M.shape[-1]))
                if len(diag) == n_dof:
                    jd = diag
                elif len(diag) == n_dof + 6:
                    jd = diag[6:] if np.allclose(diag[:3], diag[0], rtol=0.05) else diag[:n_dof]
                else:
                    raise ValueError(f"mass matrix diagonal {diag.shape} vs {n_dof} DOF")
                self._am_inertia = np.maximum(self._to_canonical(jd), 1e-6)
                if not self._am_logged:
                    self._am_logged = True
                    self._logger.info(f"[{self._name}] servo friction: stopping-torque form with the PhysX mass-matrix "
                                      f"diagonal (M {tuple(M.shape)}): {[round(float(v), 5) for v in self._am_inertia]}"
                                      + (f"; ext estimate {[round(float(v), 3) for v in ext]}" if ext is not None else "; no ext estimate"))
            except Exception as e:
                self._am_inertia_ok = False
                self._am_inertia = None
                self._logger.info(f"[{self._name}] servo friction: dissipative band form (no mass matrix: {e})")
        elif not self._am_logged:
            self._am_logged = True
            self._logger.info(f"[{self._name}] servo friction: dissipative band form (stiction band "
                              f"{[s.stiction_velocity for s, _ in am.groups]} rad/s)"
                              + (f"; ext estimate {[round(float(v), 3) for v in ext]}" if ext is not None else "; no ext estimate"))
        tau = am.compute(pos_t, q, dq, enable=(mode == protocol.MODE_POSITION), ext=ext, inertia=self._am_inertia)
        damp = am.dampings().astype(np.float32)
        kp[m] = 0.0
        kd[m] = damp[m]                 # implicit: PhysX applies -kd dq inside the solver
        vel_t[m] = 0.0
        eff[m] = tau[m].astype(np.float32)
        self._am_tau = am.tau_motor.copy()

    # -- tensor reads and writes -------------------------------------------------------
    def _batch_view(self, art):
        """The batch when it carries this robot and reports its joint state, else None (per-robot path)."""
        batch = self._batch
        if batch is None or not batch.ok:
            return None
        if self._batch_gen != batch.generation:
            self._batch_gen = batch.generation
            self._batch_row = batch.row(self._path)
            self._batch_check_left = 0                # verify before the first use of a (re)bound batch
            self._batch_verified = False
            if self._batch_row < 0:
                self._logger.info(f"[{self._name}] not in the articulation batch; own tensor view")
        if self._batch_row < 0 or self._dof_count(art) <= 0:
            return None
        step = batch.sampled_step
        if step == self._batch_step_seen:
            return batch                              # synchronize() asks twice per step; resolve once
        self._batch_step_seen = step
        if self._batch_check_left > 0:
            self._batch_check_left -= 1
            return batch
        if not self._verify_batch_row(art, batch):
            return None
        self._batch_check_left = _BATCH_RECHECK_STEPS
        return batch

    @staticmethod
    def _dof_count(art) -> int:
        """This robot's PhysX DOF count: how much of its (padded) batch row belongs to it."""
        n = getattr(art, "num_dof", None)
        if n:
            return int(n)
        return len(list(getattr(art, "dof_names", []) or []))

    def _verify_batch_row(self, art, batch) -> bool:
        """This robot's own view against its row of the batch. Anything but an exact match (a wrong row, a DOF order
        that is not PhysX's, a stale buffer) drops the robot back to its own view instead of misdriving it."""
        try:
            pv = self._raw_view(art)
            if pv is not None:
                q = np.asarray(pv.get_dof_positions()[0], dtype=np.float64)
            else:
                q = np.asarray(art.get_joint_positions()[0], dtype=np.float64)
        except Exception as e:
            self._logger.debug(f"[{self._name}] batch verification read failed: {e}")
            return False                              # cannot verify now: this step goes per-robot
        if batch.matches(self._batch_row, q):
            if not self._batch_verified:
                self._batch_verified = True
                self._logger.debug(f"[{self._name}] joint state and targets through the articulation batch "
                                   f"(row {self._batch_row}, generation {batch.generation})")
            return True
        self._logger.warning(f"[{self._name}] articulation batch row {self._batch_row} does not report this "
                             f"robot's joint state; falling back to its own tensor view")
        self._batch_row = -1
        return False

    def _raw_view(self, art):
        """The numpy-backed omni.physics.tensors ArticulationView behind `art`, or None (wrapper path).

        Per-step joint state and drive targets go straight through the view Isaac's Articulation wrapper owns:
        each wrapper call resolves index tensors, copies and re-indexes, and a robot makes several per step."""
        pv = getattr(art, "_physics_view", None)
        if pv is None:
            return None
        if self._raw_ok is None:
            try:
                probe = pv.get_dof_positions()
                self._raw_ok = isinstance(probe, np.ndarray) and probe.ndim == 2 and probe.shape[0] >= 1
            except Exception as e:
                self._raw_ok = False
                self._logger.info(f"[{self._name}] raw tensor view unusable ({type(e).__name__}: {e}); wrapper path")
            if self._raw_ok:
                self._logger.debug(f"[{self._name}] joint state and targets through the raw tensor view")
            else:
                self._logger.info(f"[{self._name}] tensor view is not numpy-backed; wrapper path")
        return pv if self._raw_ok else None

    def _read_dq(self, art, pv, batch=None) -> np.ndarray:
        if batch is not None:
            return batch.velocities(self._batch_row, self._dof_count(art))
        if pv is not None:
            return np.array(pv.get_dof_velocities()[0], dtype=np.float64)
        return np.asarray(art.get_joint_velocities()[0], dtype=np.float64)

    def _read_measured_efforts(self, art, batch) -> np.ndarray:
        """Measured joint efforts in PhysX DOF order: this robot's row of the batch (one backend call for every
        robot together), else this robot's own wrapper call."""
        if batch is not None:
            vals = batch.projected_joint_forces(self._batch_row, self._dof_count(art))
            if vals is not None:
                return np.asarray(vals, dtype=np.float64)
        return np.asarray(art.get_measured_joint_efforts()[0], dtype=np.float64)

    def _write_gains(self, art, pv, batch, kp_dof: np.ndarray, kd_dof: np.ndarray) -> None:
        """Drive gains: staged in the batch, or written through the raw view / the wrapper."""
        if batch is not None:
            batch.stage_gains(self._batch_row, kp_dof.reshape(-1), kd_dof.reshape(-1))
        elif pv is not None:
            pv.set_dof_stiffnesses(np.ascontiguousarray(kp_dof, dtype=np.float32), _IDX0)
            pv.set_dof_dampings(np.ascontiguousarray(kd_dof, dtype=np.float32), _IDX0)
        else:
            art.set_gains(kp_dof, kd_dof)

    def _write_targets(self, art, pv, batch, pos_dof: np.ndarray, vel_dof: np.ndarray, eff_dof,
                       clear_eff: bool = False) -> None:
        """Drive targets: staged in the batch, or applied through the raw view / the wrapper. `clear_eff` zeroes a
        feed-forward torque this robot no longer asks for."""
        if batch is not None:
            batch.stage_targets(self._batch_row, pos_dof.reshape(-1), vel_dof.reshape(-1),
                                eff_dof.reshape(-1) if eff_dof is not None else None)
            if clear_eff:
                batch.clear_efforts(self._batch_row)
        elif pv is not None:
            pv.set_dof_position_targets(np.ascontiguousarray(pos_dof, dtype=np.float32), _IDX0)
            pv.set_dof_velocity_targets(np.ascontiguousarray(vel_dof, dtype=np.float32), _IDX0)
            if eff_dof is not None:
                pv.set_dof_actuation_forces(np.ascontiguousarray(eff_dof, dtype=np.float32), _IDX0)
            elif clear_eff:
                pv.set_dof_actuation_forces(np.zeros_like(pos_dof, dtype=np.float32), _IDX0)
        else:
            art.apply_action(ArticulationActions(
                joint_positions=pos_dof, joint_velocities=vel_dof,
                joint_efforts=(np.zeros_like(pos_dof, dtype=np.float32) if (eff_dof is None and clear_eff) else eff_dof)))

    # -- published effort -----------------------------------------------------------------
    def _estimate_drive_effort(self, n, q, dq_m, fmax, tau):
        kp_d, kd_d, pos_t, vel_t, eff = self._last_drive
        m = (np.asarray(kp_d[:n], dtype=np.float64) > 0.0) if fmax is not None else np.zeros(n, dtype=bool)
        tau_d = (np.asarray(kp_d[:n], dtype=np.float64) * (np.asarray(pos_t[:n], dtype=np.float64) - q[:n])
                 + np.asarray(kd_d[:n], dtype=np.float64) * (np.asarray(vel_t[:n], dtype=np.float64) - dq_m[:n])
                 + np.asarray(eff[:n], dtype=np.float64))
        if fmax is not None:
            tau_d = np.clip(tau_d, -fmax[:n], fmax[:n])
        return np.where(m, tau_d, tau)

    def _measured_effort_needed(self, n: int) -> bool:
        """False only when the measured joint efforts would be discarded for every DOF: a drive is resolved and has a
        force limit, and each DOF is either PD-driven (kp > 0, drive estimate) or under a servo / capped model. A
        kp = 0 DOF (IDLE, limp IMPEDANCE, VELOCITY, TORQUE) publishes the measured reaction torque, so it keeps the
        read."""
        if self._last_drive is None:
            return True
        fmax = self._fmax_canon
        if fmax is None or fmax is False:
            return True            # no drive force limit (or not looked up yet): the drive estimate is not published
        try:
            kp_d = np.asarray(self._last_drive[0][:n], dtype=np.float64)
            if len(kp_d) != n:
                return True
            covered = kp_d > 0.0
            if self._am is not None and self._am_tau is not None:
                covered = covered | np.asarray(self._am.mask[:n], dtype=bool)
            capped = self._capped_tau
            if capped is not None and self._capped_position is not None:
                covered = covered | np.asarray(capped[0][:n], dtype=bool)
            return not bool(covered.all())
        except Exception:
            return True

    def _effort_fallback(self, art, n: int, tau: np.ndarray, measured_read: bool, err: Exception) -> np.ndarray:
        """Efforts published when the drive estimate raised: the measured reaction torque — read here when the step
        skipped it — or zeros when that read fails too. Warns once, naming what is published."""
        what = "measured efforts"
        if not measured_read:
            try:
                tau = self._to_canonical(np.asarray(art.get_measured_joint_efforts()[0], dtype=np.float64))
            except Exception:
                tau = np.zeros(n, dtype=np.float64)
                what = "zeros (measured read failed)"
        if not self._effort_warned:
            self._effort_warned = True
            self._logger.warning(f"[{self._name}] drive-effort estimate unavailable ({err}); publishing {what}")
        return tau

    def _max_efforts(self, art):
        """Drive force limit per DOF (canonical order), looked up once: the USD drive maxForce of each joint first,
        PhysX's get_max_efforts() as the fallback. None when neither is usable; then no drive estimate is published
        at all (see _published_effort)."""
        if self._fmax_canon is None:
            source = None
            try:
                names = list(getattr(art, "dof_names", []) or [])
                joints = self._joints_by_name()
                fm = np.zeros(len(names), dtype=np.float64)
                for i, name in enumerate(names):
                    joint = joints.get(name)
                    if joint is None:
                        continue
                    angular = joint.GetTypeName() == "PhysicsRevoluteJoint"
                    attr = joint.GetAttribute("drive:angular:physics:maxForce" if angular else "drive:linear:physics:maxForce")
                    if attr and attr.HasAuthoredValue():
                        fm[i] = float(attr.Get())
                if fm.size and np.all(np.isfinite(fm)) and np.all(fm > 0):
                    self._fmax_canon, source = self._to_canonical(fm), "USD drive maxForce"
            except Exception:
                pass
            if self._fmax_canon is None:
                try:
                    fm = np.asarray(art.get_max_efforts()[0], dtype=np.float64)
                    if fm.size and np.all(np.isfinite(fm)) and np.all(fm > 0) and np.all(fm < _EFFORT_LIMIT_SENTINEL):
                        self._fmax_canon, source = self._to_canonical(fm), "PhysX get_max_efforts"
                    else:
                        self._fmax_canon = False
                except Exception:
                    self._fmax_canon = False
            if self._fmax_canon is False:
                self._logger.warning(f"[{self._name}] effort: no usable drive maxForce (USD or PhysX); publishing the "
                                     f"measured joint reaction torque, not the unbounded drive estimate")
            else:
                self._logger.info(f"[{self._name}] effort clamp from {source}: "
                                  f"{self._fmax_canon.min():.3g}..{self._fmax_canon.max():.3g} Nm")
        return None if self._fmax_canon is False else self._fmax_canon

    # -- teleport and respawn ----------------------------------------------------------------
    def _apply_teleport(self) -> None:
        """Teleport only the DOFs flagged 0xFF (gateway windows teleport independently); every other DOF — another
        gateway's joints, idle casters — keeps its pose."""
        n = self._n
        m = self._tp_mask[:n]
        try:
            pos_full = np.asarray(self._art.get_joint_positions()[0], dtype=np.float32)
            vel_full = np.asarray(self._art.get_joint_velocities()[0], dtype=np.float32)
            pos_c = self._to_canonical(pos_full)
            vel_c = self._to_canonical(vel_full)
        except Exception:
            pos_full = vel_full = None
            pos_c = np.zeros(n, dtype=np.float32)
            vel_c = np.zeros(n, dtype=np.float32)
        pos_c[m] = self._tp_q[:n][m]
        vel_c[m] = 0.0
        n_dof = self._n_dof or n
        try:
            self._art.set_joint_positions(self._to_dof(pos_c, base=pos_full).reshape(1, n_dof))
            self._art.set_joint_velocities(self._to_dof(vel_c, base=vel_full).reshape(1, n_dof))
        except Exception as e:
            self._logger.warning(f"[{self._name}] teleport failed: {e}")
        # the teleported DOFs drop their drive targets (their gateway sends again within a cycle); other gateways'
        # DOFs keep driving
        self._mode[:n][m] = protocol.MODE_IDLE
        self._tp_mask[:] = False
        self._last_kp = self._last_kd = None
        self._cmd_dirty = True
        if self._am is not None:
            self._am.reset()

    def _clean_respawn(self, art) -> None:
        """Put the articulation back at its authored spawn state: the PhysX root link at the world pose USD authors
        for it (see _spawn_root_pose), every joint at its PhysicsJointStateAPI position, all velocities zero.

        Runs two physics steps after every (re)bind and after a same-stage physics re-init: a timeline stop -> play
        re-places the links but leaves the joints in their pre-reset configuration for several steps, so a robot
        that had fallen would come back folded. RC_ISAAC_RESPAWN_AT_BIND=0 turns it off."""
        try:
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(self._path) if stage else None
            if root is None or not root.IsValid():
                return
            names = list(getattr(art, "dof_names", []) or [])
            if not names:
                return
            joints = self._joints_by_name()
            q0 = np.zeros(len(names), dtype=np.float32)
            authored = 0
            for i, name in enumerate(names):
                joint = joints.get(name)
                if joint is None:
                    continue
                angular = joint.GetTypeName() == "PhysicsRevoluteJoint"
                attr = joint.GetAttribute("state:angular:physics:position" if angular else "state:linear:physics:position")
                if attr and attr.HasAuthoredValue():
                    value = float(attr.Get())
                    q0[i] = np.radians(value) if angular else value
                    authored += 1
            # The base: the PhysX root LINK at its USD world pose, composed through every parent Xform and taken at
            # the root body itself — not the articulation prim's local transform, which ignores parent Xforms (a
            # robot under a group prim would land at the origin) and the body's own offset inside the prim.
            spawn = None
            try:
                spawn = self._spawn_root_pose(art, stage)
            except Exception as e:
                self._logger.warning(f"[{self._name}] spawn pose lookup failed, base left in place: "
                                     f"{type(e).__name__}: {e}")
            if spawn is not None:
                art.set_world_poses(positions=spawn.pos, orientations=spawn.quat.reshape(1, 4))
            n_dof = len(names)
            art.set_joint_velocities(np.zeros((1, n_dof), dtype=np.float32))
            art.set_joint_positions(q0.reshape(1, n_dof))
            try:
                art.set_velocities(np.zeros((1, 6), dtype=np.float32))
            except Exception:
                pass
            # Hold the placed pose: the command in force is whatever the gateway sent before the timeline stopped (a
            # mid-gait pose, say), and the gateway sends a fresh one only after it sees the first new state. Driving
            # the re-placed joints toward that stale command would kick them; target the spawn pose until then.
            self._hold_spawn_pose(q0)
            if spawn is not None:
                where = (f"base = {spawn.source} {spawn.link} at its USD world pose "
                         f"t={[round(float(v), 4) for v in spawn.pos[0]]} q={[round(float(v), 4) for v in spawn.quat]}"
                         + (f" ({spawn.moved:.4f} m from where PhysX had it)" if spawn.moved is not None else ""))
            else:
                where = "base left in place (no spawn pose)"
            self._logger.info(f"[{self._name}] clean respawn: {where}, {authored}/{n_dof} joints at the USD initial "
                              f"state, velocities zero, drive targets = that pose")
        except Exception as e:
            self._logger.warning(f"[{self._name}] clean respawn skipped: {type(e).__name__}: {e}")

    def _spawn_root_pose(self, art, stage) -> "Optional[SpawnPose]":
        """The world pose USD authors for this articulation's PhysX root link — what set_world_poses() (Isaac:
        ArticulationView.set_root_transforms, the root LINK, not the prim) must be given to put the robot back at
        its spawn. Composed through every parent Xform with UsdGeom.XformCache and taken at the link the view reports
        as parentless; when the view does not say, the first rigid body under the prim, else the prim itself. None
        when nothing finite comes out (the caller then leaves the base where it is)."""
        pv = getattr(art, "_physics_view", None)
        link, source = self._root_link_path(art, stage)
        pose = self._usd_world_pose(stage, link)
        if pose is None:
            return None
        pos, quat = pose
        moved = None
        if pv is not None:
            try:
                rt = np.asarray(pv.get_root_transforms(), dtype=np.float64).reshape(-1, 7)[0]
                if np.all(np.isfinite(rt[:3])):
                    moved = float(np.linalg.norm(rt[:3] - pos[0].astype(np.float64)))
            except Exception:
                moved = None
        return SpawnPose(pos=pos, quat=quat, link=link, source=source, moved=moved)

    def _root_link_path(self, art, stage=None) -> "Tuple[str, str]":
        """(prim path, source) of this articulation's PhysX root link: the parentless link of its tensor view, else
        the first rigid body under the prim (`stage`, default the open stage), else the articulation prim itself.
        The one definition behind the clean respawn's spawn pose and the root_pose frame."""
        pv = getattr(art, "_physics_view", None)
        link = self._physx_root_link_path(pv, self._path) if pv is not None else None
        if link is not None:
            return link, "root link"
        if stage is None:
            root = self._root_prim()
            stage = root.GetStage() if root else None
        link = self._first_rigid_body_path(stage, self._path)
        if link is not None:
            return link, "first rigid body"
        return self._path, "articulation prim"

    @staticmethod
    def _physx_root_link_path(pv, root_path: str) -> "Optional[str]":
        """Prim path of the PhysX root link of the (single) articulation behind the raw view `pv`: the link whose
        metatype parent is empty, else the view's first link. None when the view does not expose its links."""
        try:
            links = pv.link_paths
        except Exception:
            return None
        if links is None:
            return None
        try:
            links = list(links)
            if links and isinstance(links[0], (list, tuple)):
                links = list(links[0])                # (count, links): one articulation per raw view
            links = [str(p) for p in links if p]
        except Exception:
            return None
        if not links:
            return None
        idx = 0
        try:
            parents = list(getattr(pv.shared_metatype, "link_parents", None) or [])
            for i, parent in enumerate(parents[:len(links)]):
                if not parent:
                    idx = i
                    break
        except Exception:
            pass
        return links[idx]

    @staticmethod
    def _first_rigid_body_path(stage, root_path: str) -> "Optional[str]":
        """The first prim with PhysicsRigidBodyAPI under `root_path` in authoring order (a fallback root link)."""
        try:
            from pxr import Usd, UsdPhysics
            root = stage.GetPrimAtPath(root_path) if stage is not None else None
            if not root or not root.IsValid():
                return None
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(prim.GetPath())
        except Exception:
            return None
        return None

    @staticmethod
    def _usd_world_pose(stage, path: str):
        """((1, 3) float32 position, (4,) float32 quaternion wxyz) of `path` in the world frame at the default time
        code, scale and shear factored out of the rotation; None when the prim is missing or the result not finite."""
        from pxr import Gf, Usd, UsdGeom
        prim = stage.GetPrimAtPath(path) if stage is not None else None
        if prim is None or not prim.IsValid():
            return None
        m = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        xf = Gf.Transform(m)
        t = xf.GetTranslation()
        rq = xf.GetRotation().GetQuat()
        im = rq.GetImaginary()
        pos = np.array([[float(t[0]), float(t[1]), float(t[2])]], dtype=np.float32)
        quat = np.array([float(rq.GetReal()), float(im[0]), float(im[1]), float(im[2])], dtype=np.float32)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return None
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        return pos, quat / norm

    def _hold_spawn_pose(self, q0_dof: np.ndarray) -> None:
        """After a clean respawn: the command in force targets the re-placed joints (DOF order q0) until the
        gateway's next command replaces it."""
        n = self._n
        if self._q_d is None or not n:
            return
        q0_c = np.asarray(self._to_canonical(np.asarray(q0_dof, dtype=np.float32)), dtype=np.float32)
        self._q_d[:n] = q0_c[:n]
        self._dq_d[:n] = 0.0
        self._tau_d[:n] = 0.0
        if self._latched is not None:
            self._latched = (self._latched[0], self._q_d.copy(), self._dq_d.copy(), self._tau_d.copy(),
                             self._latched[4], self._latched[5])
        self._cmd_dirty = True
        self._last_drive = None

    # -- physics re-initialization (timeline stop -> play on the same stage) ------------------
    def _check_physics_reinit(self, art) -> None:
        pv = getattr(art, "_physics_view", None)
        fresh = pv is not None and self._phys_view is not None and pv is not self._phys_view
        if pv is not None:
            self._phys_view = pv
        if fresh or self._reinit_pending:
            self._reinit_pending = False
            self._on_physics_reinit(art, "physics view re-created" if fresh else "timeline resumed")

    def _on_physics_reinit(self, art, why: str) -> None:
        """PhysX re-parsed the USD under a device that was kept: every value written through the tensor API is back
        at its USD default. Forget the gain cache so the next drive writes it again, put the servo model's armature
        and velocity clamp back, reset the model's state (delay line, friction memory) and drop the stale targets —
        the robot stands at its spawn pose again and its gateway sends the live command within a cycle (as after a
        teleport)."""
        self._logger.info(f"[{self._name}] {why}: re-applying drive gains and servo-model parameters, "
                          f"dropping stale targets")
        if self._respawn_at_bind:
            self._respawn_left = 2                   # place the robot again once the new view is live
        self._last_kp = self._last_kd = None
        self._logged_apply = False
        self._latched = None
        self._cmd_dirty = True
        self._raw_ok = None                          # probe the new view again
        self._batch_gen = -1                         # ... and re-verify this robot's row of the new batch
        self._root_link = None                       # root_pose: resolve the root link against the new view
        self._root_row_gen = -1
        if self._mode is not None:
            self._mode[:] = protocol.MODE_IDLE
        if self._am is not None:
            self._am.reset()
            if self._am_tau is not None:
                self._am_tau[:] = 0.0
            self._am_inertia = None
            self._am_step = 0
            self._apply_model_physx_params(art)

    # -- root_pose (device/root_pose.py) ------------------------------------------------------
    def _root_pose_frame(self, sim_time: float):
        """This step's root_pose payload, or None: not due yet, turned off, or no pose to read this step (the next
        step tries again: only a published frame moves the rate limiter)."""
        rate = self._root_rate
        if not rate.due(sim_time):
            return None
        state = self._root_state()
        if state is None:
            return None
        pos, quat, lin_vel, ang_vel = state
        seq = self._root_seq + 1
        payload = encode_root_pose(sim_time, seq, pos, quat, lin_vel, ang_vel)
        rate.mark(sim_time)                      # a non-finite pose also waits a period before it is checked again
        if payload is None:
            self._root_skips += 1
            if self._root_skips == 1 or self._root_skips % 1000 == 0:
                self._logger.warning(f"[{self._name}] root link pose is not finite, root_pose not published "
                                     f"({self._root_skips} so far)")
            return None
        self._root_seq = seq
        return payload

    def _root_state(self):
        """(pos (3,), quat wxyz (4,), lin_vel (3,), ang_vel (3,)) float64 of the PhysX root link, world frame, or None.

        The link view already holds every tracked link's transform and velocity for this step (the moderator sampled
        it before the devices ran), so the root link's row costs no backend call. A robot whose root link is not in
        that view reads the articulation view's root transform instead, at the same rate. A link view that has not
        sampled this step (physics warming up, a rebind) means no pose now, not a fallback read."""
        view = self._link_view
        if view is not None:
            if not view.ok:
                return None
            row = self._root_view_row(view)
            if row is None:
                return None
            if row >= 0:
                p, q = view.pose_wxyz(row)
                v6 = view.velocity(row)
                self._note_root_source("link view")
                return p, q, v6[:3], v6[3:6]
        return self._root_state_from_articulation()

    def _root_view_row(self, view):
        """Row of the PhysX root link in the link view (-1 = not there), looked up once per view generation; None
        while the root link cannot be resolved yet (no physics view)."""
        if self._root_row_gen == view.generation:
            return self._root_row
        link = self._resolve_root_link()
        if link is None:
            return None
        self._root_row_gen = view.generation
        self._root_row = view.row(link[0])
        if self._root_row < 0:
            self._logger.debug(f"[{self._name}] root link {link[0]} ({link[1]}) is not in the link view; root_pose "
                               f"reads the articulation view's root transform")
        return self._root_row

    def _resolve_root_link(self):
        """(path, source) of the PhysX root link (see _root_link_path), cached; None while the physics view is missing."""
        if self._root_link is None:
            art = self._art
            if getattr(art, "_physics_view", None) is None:
                return None
            self._root_link = self._root_link_path(art)
        return self._root_link

    def _root_state_from_articulation(self):
        """Root link state from the articulation tensor view: this robot's row of the batch (one backend call for
        every robot that asks in this step), else its own raw view. None when neither can say."""
        art = self._art
        xf = v6 = None
        batch = self._batch
        try:
            if (batch is not None and batch.ok and self._batch_verified
                    and self._batch_row >= 0 and self._batch_gen == batch.generation):
                xf = batch.root_transform(self._batch_row)
                v6 = batch.root_velocity(self._batch_row)
            if xf is None or v6 is None:
                pv = self._raw_view(art)
                if pv is None:
                    return None
                xf = pv.get_root_transforms()
                v6 = pv.get_root_velocities()
            xf = np.asarray(xf, dtype=np.float64).reshape(-1)
            v6 = np.asarray(v6, dtype=np.float64).reshape(-1)
        except Exception as e:
            self._logger.debug(f"[{self._name}] root transform read failed: {type(e).__name__}: {e}")
            return None
        if xf.size < 7 or v6.size < 6:
            return None
        self._note_root_source("articulation view")
        # tensor layout x y z qx qy qz qw -> quaternion w x y z
        return xf[:3].copy(), np.array((xf[6], xf[3], xf[4], xf[5])), v6[:3].copy(), v6[3:6].copy()

    def _note_root_source(self, source: str) -> None:
        if self._root_source != source:
            self._root_source = source
            self._logger.debug(f"[{self._name}] root_pose from the {source} "
                               f"(every {self._root_rate.period:.3f} s of sim time at most)")

    # -- robot metadata (rc:* attributes) ----------------------------------------------------
    def _root_prim(self):
        try:
            stage = omni.usd.get_context().get_stage()
            return stage.GetPrimAtPath(self._path) if stage else None
        except Exception:
            return None

    def _ensure_robot_meta(self, n: int) -> None:
        """Read the robot's rc:* attributes once the physics view is live. Everything is optional: a robot without
        them gets the default gains and publishes no imu_state or contact_forces."""
        if self._meta_loaded:
            return
        self._meta_loaded = True
        root = self._root_prim()
        if not root:
            return
        self._read_position_gains(root, n)
        try:
            self._load_actuator_models(root, n)
        except Exception as e:
            # nothing half-attached: a model whose setup failed must never be stepped
            self._detach_actuator_models()
            self._logger.warning(f"[{self._name}] actuator model setup failed: {type(e).__name__}: {e} "
                                 f"-> PhysX PD drive on every DOF")
        self._read_capped_position(root, n)
        self._read_contact_force_links(root)
        self._read_imu(root)

    def _read_position_gains(self, root, n: int) -> None:
        """rc:positionKp / rc:positionKd: POSITION-mode gains instead of the stiff defaults (small servo robots)."""
        kp = _attr_value(root, "rc:positionKp")
        kd = _attr_value(root, "rc:positionKd")
        if kp is not None and kd is not None and len(kp) == n and len(kd) == n:
            self._usd_kp = np.asarray(list(kp), dtype=np.float32)
            self._usd_kd = np.asarray(list(kd), dtype=np.float32)
            self._usd_vel_ff = bool(_attr_value(root, "rc:positionVelocityFeedforward") or False)
            self._logger.info(f"[{self._name}] POSITION gains from rc:positionKp/Kd "
                              f"(kp {self._usd_kp.min():.3g}..{self._usd_kp.max():.3g}, "
                              f"kd {self._usd_kd.min():.3g}..{self._usd_kd.max():.3g})")
        elif kp is not None:
            self._logger.warning(f"[{self._name}] rc:positionKp has {len(kp)} entries but the articulation has {n} "
                                 f"actuators; ignored, the default gains apply")

    def _read_capped_position(self, root, n: int) -> None:
        """rc:positionDriveModel = "capped_p_passive" with rc:positionTorqueLimits / rc:positionFrictionLoss."""
        self._capped_position = None
        if _attr_value(root, "rc:positionDriveModel") != "capped_p_passive":
            return
        try:
            if self._am is not None:
                raise ValueError("capped_p_passive cannot be combined with a servo actuator model")
            limits = np.asarray(_attr_value(root, "rc:positionTorqueLimits"), dtype=np.float64)
            friction = np.asarray(_attr_value(root, "rc:positionFrictionLoss"), dtype=np.float64)
            physical = self._to_canonical(np.asarray(self._art.get_max_efforts()[0], dtype=np.float64))
            if limits.shape != (n,) or friction.shape != (n,):
                raise ValueError("rc:positionTorqueLimits and rc:positionFrictionLoss need one value per actuator")
            if not np.isfinite(np.r_[limits, friction, physical]).all() or np.any(limits <= 0) or np.any(friction < 0):
                raise ValueError("invalid torque limits or friction")
            if np.any(limits > physical + 1e-5):
                raise ValueError("the torque limits exceed the PhysX drive limits")
            self._capped_position = dict(limits=limits, friction=friction)
            self._logger.info(f"[{self._name}] capped-P position drive; the drive's total effort limits are kept")
        except Exception as exc:
            self._logger.error(f"[{self._name}] capped-P position drive rejected: {exc}")

    def _read_contact_force_links(self, root) -> None:
        self._contact_force_paths = None
        links = _attr_value(root, "rc:contactForceLinks")
        if not links:
            return
        try:
            self._contact_force_paths = self._contact_force_links(root, links)
        except Exception as exc:
            self._logger.error(f"[{self._name}] rc:contactForceLinks rejected: {type(exc).__name__}: {exc}")

    def _contact_force_links(self, root, value) -> "List[str]":
        """Prim paths of rc:contactForceLinks; authors PhysxContactReportAPI where it is missing."""
        from pxr import PhysxSchema, UsdPhysics
        names = [str(v).strip() for v in (value.strip("[]").split(",") if isinstance(value, str) else value)]
        names = [n for n in names if n]
        if not 0 < len(names) <= protocol.CONTACT_FORCES_MAX:
            raise ValueError(f"expected 1..{protocol.CONTACT_FORCES_MAX} links, got {len(names)}")
        stage = root.GetStage()
        paths, authored = [], []
        for name in names:
            path = f"{self._path}/links/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"{path} is not a rigid body")
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                authored.append(name)
            paths.append(path)
        if authored:
            # PhysX parses contact reporting at timeline play: these links report zeros until the next play
            self._logger.warning(f"[{self._name}] PhysxContactReportAPI authored at bind on {authored}: "
                                 f"contact forces read 0 until the next timeline play (author it in the world USD)")
        self._logger.info(f"[{self._name}] contact_forces enabled for {names}")
        return paths

    def _read_imu(self, root) -> None:
        """rc:imuLink (+ rc:imuOffset, rc:imuAccelTau) and the foot contacts (rc:contactLinks, rc:contactOffsets,
        rc:contactHeight, rc:groundZ)."""
        imu_link = _attr_value(root, "rc:imuLink")
        if not imu_link:
            self._imu = False
            return
        try:
            imu_off = np.asarray(list(_attr_value(root, "rc:imuOffset") or (0.0, 0.0, 0.0)), dtype=np.float64)
            c_links = list(_attr_value(root, "rc:contactLinks") or [])
            c_offs = [np.asarray(list(v), dtype=np.float64) for v in (_attr_value(root, "rc:contactOffsets") or [])]
            c_h = float(_attr_value(root, "rc:contactHeight") or 0.01)
            override = config.contact_height_override()
            if override is not None:
                c_h = override
            ground_z = float(_attr_value(root, "rc:groundZ") or 0.0)
            imu_path = f"{self._path}/links/{imu_link}"
            imu_prim = RigidPrim(imu_path, name=f"{self._name}_imu")
            # PhysX reports velocities at the centre of mass and poses at the link origin: the rigid body's COM
            # (fixed child masses merged in) moves the velocity to the IMU point
            imu_prim.initialize()
            com_local = np.asarray(imu_prim.get_coms()[0], dtype=np.float64).reshape(-1)[:3]
            if com_local.shape != (3,) or not np.isfinite(com_local).all():
                raise ValueError("invalid IMU rigid-body centre of mass")
            contacts = []
            contact_paths = []
            for i, link in enumerate(c_links):
                off = c_offs[i] if i < len(c_offs) else np.zeros(3)
                contact_paths.append(f"{self._path}/links/{link}")
                contacts.append((RigidPrim(contact_paths[-1], name=f"{self._name}_c{i}"), off))
            self._imu = {
                "prim": imu_prim, "offset": imu_off, "contacts": contacts,
                "com_local": com_local,
                "path": imu_path, "contact_paths": contact_paths,
                "contact_height": c_h, "ground_z": ground_z,
                "prev_v": None, "prev_t": None,
                "accel_tau": float(_attr_value(root, "rc:imuAccelTau") or 0.02),
            }
            self._imu_rows = None
            self._logger.info(f"[{self._name}] imu_state enabled: link '{imu_link}' offset {imu_off.tolist()}, "
                              f"contacts {c_links} (height < {c_h} m above z={ground_z})")
        except Exception as e:
            self._logger.warning(f"[{self._name}] rc:imuLink present but the IMU could not be set up: {e}")
            self._imu = False

    def _load_actuator_models(self, root, n: int) -> None:
        """Build the per-DOF servo models from rc:actuator:* (joint prims + root defaults). No-op for robots without
        them. Joints whose params fail validation keep the PD drive (read_actuator_specs names each in a WARNING); the
        model set is fully built, described and checked BEFORE it is attached, so a failure leaves no model behind."""
        joints = self._joint_prims()
        if len(joints) != n:
            try:
                bam = [p for p in self._all_joint_prims()
                       if str(_attr_value(p, ATTR_MODEL) or "").strip().lower() == MODEL_BAM]
            except Exception:
                bam = []
            if bam:
                self._logger.warning(
                    f"[{self._name}] BAM servo models skipped: {len(joints)} actuated USD joints vs {n} articulation "
                    f"DOF, so rc:actuator:* cannot be mapped onto the DOFs ({len(bam)} joint(s) say \"bam\") "
                    f"-> PhysX PD drive on every DOF")
            return
        specs = read_actuator_specs(root, joints, log=self._logger, owner=self._name)
        if not any(specs):
            return
        try:
            dt = float(self._sim_context.get_physics_dt())
        except Exception:
            dt = 1.0 / 200.0
        am = ActuatorModelSet(specs, dt)
        for idx, reason in am.rejected:     # read_actuator_specs validates too; this is the last line of defence
            self._logger.warning(f"[{self._name}] BAM servo model rejected on DOF {idx}: {reason} "
                                 f"-> PhysX PD drive on those DOF")
        if not am.active:
            return
        am.set_implicit_damping(True)      # back-EMF + viscous friction as PhysX drive damping (implicit)
        # everything that divides by the model parameters runs here, before the attach
        damp = am.dampings()
        if not np.all(np.isfinite(damp)):
            raise ValueError(f"non-finite implicit damping {damp.tolist()}")
        desc = am.describe()
        self._am_joints = joints
        self._am_tau = np.zeros(n, dtype=np.float64)
        self._am = am
        try:
            self._apply_model_physx_params(self._art)
        except Exception:
            self._detach_actuator_models()
            raise
        self._logger.info(f"[{self._name}] actuator models on {int(am.mask.sum())}/{n} DOF "
                          f"(physics dt {dt * 1e3:.2f} ms): {desc}")
        if dt > self.AM_MAX_VALIDATED_DT * (1.0 + 1e-6):
            self._logger.warning(
                f"[{self._name}] BAM servo models run with a {dt * 1e3:.2f} ms physics step (> "
                f"{self.AM_MAX_VALIDATED_DT * 1e3:.0f} ms): they were validated at 200 Hz and chatter at the Nyquist "
                f"rate at 100 Hz; raise the world's physics rate to 200 Hz or more (behaviour unchanged)")

    def _detach_actuator_models(self) -> None:
        """Drop the servo models: every DOF back on the PhysX drive."""
        self._am = None
        self._am_tau = None
        self._am_joints = None
        self._am_inertia = None

    def _apply_model_physx_params(self, art) -> None:
        """Push the servo model's PhysX-side parameters through the tensor API: the rotor armature (the model owns
        the identified value) and, per DOF that opts in, the lifted joint velocity clamp. Both are runtime-only in
        PhysX, so this runs at model load and again after every physics re-initialization (_on_physics_reinit)."""
        am = self._am
        if am is None:
            return
        n = self._n
        n_dof = self._n_dof or n
        m_dof = self._to_dof(am.mask, fill=False)
        try:
            arm = am.armatures()
            cur = np.asarray(art.get_armatures()[0], dtype=np.float64).reshape(-1)
            if len(cur) == n_dof:
                new = cur.copy()
                arm_dof = self._to_dof(np.where(np.isnan(arm), 0.0, arm))
                new[m_dof] = arm_dof[m_dof]
                if not np.allclose(new, cur, atol=1e-7):
                    art.set_armatures(new.reshape(1, n_dof).astype(np.float32))
                    self._logger.info(f"[{self._name}] armature from the actuator model: "
                                      f"{[round(float(v), 5) for v in new]}")
        except Exception as e:
            self._logger.debug(f"[{self._name}] armature update skipped: {e}")
        # The PhysX joint velocity clamp is KEPT (the USD value) by default: lifted, a light link can spin up to
        # hundreds of rad/s on a contact impulse. The servo param lift_velocity_clamp = true opts in per DOF; passive
        # and non-modelled DOFs are never touched (PhysX has no getter to restore them from).
        lift = am.lift_velocity_clamp()
        if bool(lift.any()):
            try:
                from pxr import UsdPhysics
                vmax = np.full(n, 1.0e3, dtype=np.float32)
                for i, prim in enumerate(self._am_joints or []):
                    if am.mask[i] and lift[i]:
                        continue
                    v = _attr_value(prim, "physxJoint:maxJointVelocity")
                    if v is None:
                        v = 1.0e6
                    vmax[i] = np.radians(float(v)) if prim.IsA(UsdPhysics.RevoluteJoint) else float(v)
                art.set_max_joint_velocities(self._to_dof(vmax, fill=1.0e6).reshape(1, n_dof))
                self._logger.info(f"[{self._name}] joint velocity clamp lifted on DOF {np.flatnonzero(lift).tolist()} "
                                  f"(lift_velocity_clamp)")
            except Exception as e:
                self._logger.debug(f"[{self._name}] joint velocity limit update skipped: {e}")

    # -- IMU emulation -----------------------------------------------------------------------
    def _imu_view_rows(self):
        """(IMU row, [contact rows]) in the link view, or None (no view, not sampled, links not in it)."""
        view = self._link_view
        if view is None or not view.ok:
            return None
        if self._imu_rows is None or self._imu_rows_gen != view.generation:
            imu = self._imu
            rows = [view.row(imu.get("path", ""))] + [view.row(p) for p in imu.get("contact_paths", [])]
            self._imu_rows_gen = view.generation
            self._imu_rows = (rows[0], rows[1:]) if all(r >= 0 for r in rows) else False
            if self._imu_rows is False:
                self._logger.info(f"[{self._name}] IMU links not in the link view; per-prim reads")
        return self._imu_rows or None

    def _sample_imu(self) -> None:
        """Emulate a body-mounted IMU (raw gyro and accelerometer in the IMU frame, gravity included, like a BNO055)
        and two foot switches from the PhysX rigid-body state."""
        imu = self._imu
        rows = self._imu_view_rows()
        contact_poses = None
        try:
            if rows is not None:
                view = self._link_view
                p, q_wxyz = view.pose_wxyz(rows[0])
                v6 = view.velocity(rows[0])
                contact_poses = [view.pose_wxyz(r) for r in rows[1]]
            else:
                pos, quat = imu["prim"].get_world_poses()
                vel = imu["prim"].get_velocities()
                p = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
                q_wxyz = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
                v6 = np.asarray(vel, dtype=np.float64).reshape(-1)
        except Exception as e:
            self._logger.debug(f"[{self._name}] IMU sample failed: {e}")
            return
        v_w, w_w = v6[:3], v6[3:6]
        R = quat_to_matrix_wxyz(np.asarray(q_wxyz, dtype=np.float64))
        r_w = R @ imu["offset"]                       # IMU point offset, world
        p_imu = p + r_w
        r_from_com = R @ (imu["offset"] - imu.get("com_local", np.zeros(3)))
        v_imu = v_w + np.cross(w_w, r_from_com)
        try:
            t = float(self._sim_context.current_time)
        except Exception:
            t = 0.0
        a_w = np.zeros(3)
        dt_step = 0.0
        if imu["prev_v"] is not None and imu["prev_t"] is not None:
            dt_step = t - imu["prev_t"]
            if dt_step > 1e-6:
                a_w = (v_imu - imu["prev_v"]) / dt_step
        imu["prev_v"], imu["prev_t"] = v_imu, t
        # PhysX velocities jitter from step to step (contact solver), so the finite-difference acceleration is very
        # noisy. A real IMU low-passes its accelerometer: first-order filter with time constant rc:imuAccelTau
        # (default 20 ms). The gravity direction, what a policy mostly uses, is unaffected.
        tau = float(imu.get("accel_tau", 0.02))
        prev_f = imu.get("accel_filt")
        if prev_f is None or tau <= 0.0 or dt_step <= 0.0:
            a_f = a_w.copy()
        else:
            alpha = dt_step / (tau + dt_step)
            a_f = prev_f + alpha * (a_w - prev_f)
        imu["accel_filt"] = a_f
        accel_b = R.T @ (a_f - GRAVITY)               # proper acceleration, IMU frame
        gyro_b = R.T @ w_w
        contacts = []
        for i, (prim, off) in enumerate(imu["contacts"]):
            try:
                if contact_poses is not None:
                    cp, cq = contact_poses[i]
                else:
                    cp, cq = prim.get_world_poses()
                    cp = np.asarray(cp, dtype=np.float64).reshape(-1)[:3]
                    cq = np.asarray(cq, dtype=np.float64).reshape(-1)[:4]
                cR = quat_to_matrix_wxyz(np.asarray(cq, dtype=np.float64))
                sole_z = float((cp + cR @ off)[2])
                contacts.append(1 if (sole_z - imu["ground_z"]) < imu["contact_height"] else 0)
            except Exception:
                contacts.append(0)
        self._imu_seq += 1
        try:
            sim_time_ns = int(t * 1e9)
        except (TypeError, ValueError):
            sim_time_ns = 0
        quat_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        self._imu_payload = self._imu_packer.pack(
            sim_time_ns, self._imu_seq, quat_xyzw, gyro_b, accel_b, v_imu, p_imu, contacts)

    # -- actuator <-> PhysX DOF mapping -------------------------------------------------------
    def _to_canonical(self, dof_vals: np.ndarray) -> np.ndarray:
        """Gather a PhysX-DOF-ordered array into actuator order (passive DOFs dropped)."""
        p = self._perm_arr
        if p is None or len(dof_vals) != (self._n_dof or len(p)):
            return dof_vals
        return dof_vals[p]

    def _to_dof(self, canon_vals: np.ndarray, fill=0.0, base=None) -> np.ndarray:
        """Scatter an actuator-ordered array into PhysX DOF order. Passive DOFs (never commanded) take `base` (a live
        DOF-ordered array, e.g. the current positions) or `fill`."""
        p = self._perm_arr
        if p is None or len(canon_vals) != len(p):
            return canon_vals
        n_dof = self._n_dof or len(p)
        if base is not None and len(base) == n_dof:
            out = np.asarray(base).astype(canon_vals.dtype).copy()
        else:
            out = np.full(n_dof, fill, dtype=canon_vals.dtype)
        out[p] = canon_vals
        return out

    def _through_backlash(self, canon_vals: np.ndarray, dof_vals: np.ndarray) -> np.ndarray:
        """Servo + backlash-hinge value per actuator (the encoder sits on the output side of the gear play)."""
        bl = self._bl_dof
        if bl is None or len(canon_vals) != len(bl) or len(dof_vals) != (self._n_dof or 0):
            return canon_vals
        return canon_vals + np.where(bl >= 0, dof_vals[np.maximum(bl, 0)], 0.0)

    def _all_joint_prims(self):
        """Every movable joint prim under this articulation in USD authoring order."""
        try:
            from pxr import Usd
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(self._path) if stage else None
            if not root:
                return []
            return [p for p in Usd.PrimRange(root) if p.GetTypeName() in _MOVABLE_JOINTS]
        except Exception as e:
            self._logger.warning(f"[{self._name}] USD joint enumeration failed: {e}")
            return []

    def _joints_by_name(self) -> "Dict[str, Prim]":
        """This articulation's movable joint prims by name (PhysX names its DOFs after the joint prims)."""
        return {str(p.GetName()): p for p in self._all_joint_prims()}

    @staticmethod
    def _joint_is_passive(prim) -> bool:
        try:
            return bool(_attr_value(prim, "rc:passive"))
        except Exception:
            return False

    def _joint_prims(self):
        """Actuator joint prims: the movable joints that are not rc:passive, authoring order."""
        return [p for p in self._all_joint_prims() if not self._joint_is_passive(p)]

    def _usd_joint_info(self):
        """[(name, passive, backlash_of, passive_kd_rad)] for every movable joint, authoring order."""
        out = []
        for p in self._all_joint_prims():
            passive = self._joint_is_passive(p)
            backlash_of, kd = None, 0.0
            if passive:
                try:
                    backlash_of = str(_attr_value(p, "rc:backlashOf") or "") or None
                    damping = _attr_value(p, "drive:angular:physics:damping")
                    # USD angular drive gains are authored per degree; the tensor API wants per radian
                    kd = float(damping) * 180.0 / np.pi if damping else 0.0
                except Exception:
                    pass
            out.append((str(p.GetName()), passive, backlash_of, kd))
        return out

    def _ensure_perm(self, art, n_dof: int) -> None:
        """Build perm[i] = PhysX DOF index of actuator i from the USD joint order matched to PhysX's dof_names.
        Passive USD joints (rc:passive) are left out of the actuator set and remembered with their authored damping
        and, for backlash hinges, the servo they belong to. Falls back to the identity (every DOF an actuator)
        whenever the two cannot be reconciled, so a robot the bridge cannot reason about is never mis-mapped."""
        if self._perm_arr is not None and self._n_dof_perm == n_dof:
            return
        dof_names = getattr(art, "dof_names", None)
        if dof_names is None and hasattr(art, "get_dof_names"):
            try:
                dof_names = art.get_dof_names()
            except Exception:
                dof_names = None
        info = self._usd_joint_info()
        canonical = [nm for nm, passive, _, _ in info if not passive]
        passive = [(nm, bo, kd) for nm, p, bo, kd in info if p]
        perm = list(range(n_dof))
        passive_hits = []
        if dof_names is not None and len(dof_names) == n_dof and len(info) == n_dof:
            dof_index = {str(nm): i for i, nm in enumerate(dof_names)}
            try:
                perm = [dof_index[nm] for nm in canonical]
                passive_hits = [(dof_index[nm], bo, kd) for nm, bo, kd in passive]
            except KeyError as e:
                self._logger.warning(f"[{self._name}] USD joint {e} not in the PhysX DOF names {list(dof_names)}; "
                                     f"actuator i = PhysX DOF i")
                perm, passive_hits, canonical = list(range(n_dof)), [], list(dof_names)
        else:
            self._logger.info(f"[{self._name}] USD joints {canonical} cannot be matched to the PhysX DOFs "
                              f"{list(dof_names) if dof_names is not None else None}; actuator i = PhysX DOF i")
            canonical = list(dof_names) if dof_names is not None and len(dof_names) == n_dof else [str(i) for i in range(n_dof)]
        self._perm_arr = np.asarray(perm, dtype=np.int64)
        self._n = len(perm)
        self._n_dof = n_dof
        self._n_dof_perm = n_dof
        self._passive_kd = np.zeros(n_dof, dtype=np.float64)
        bl = np.full(self._n, -1, dtype=np.int64)
        canon_index = {nm: i for i, nm in enumerate(canonical)}
        for di, bo, kd in passive_hits:
            self._passive_kd[di] = kd
            if bo in canon_index:
                bl[canon_index[bo]] = di
        self._bl_dof = bl if passive_hits else None
        if passive_hits:
            self._logger.info(f"[{self._name}] {len(passive_hits)} passive DOF (rc:passive) outside the actuator set: "
                              f"{[dof_names[di] for di, _, _ in passive_hits]}; {int((bl >= 0).sum())} backlash "
                              f"pair(s), published through the hinge")
        if perm == list(range(n_dof)):
            self._logger.info(f"[{self._name}] {self._n} actuators in PhysX DOF order: {canonical}")
        else:
            self._logger.info(f"[{self._name}] actuator -> PhysX DOF remap (USD joint order): perm={perm} "
                              f"usd={canonical} physx={list(dof_names)}")

    # -- diagnostics ---------------------------------------------------------------------------
    def _log_first_drive(self, art, kp: np.ndarray, kd: np.ndarray, pos_t: np.ndarray) -> None:
        """At DEBUG, once per bind: read back what PhysX accepted of the first drive. Gains and targets pass several
        broadcast and index layers that fail quietly, and "commanded but motionless" has no other symptom."""
        if self._logged_apply:
            return
        self._logged_apply = True
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        r4 = lambda a: [round(float(v), 4) for v in np.asarray(a).ravel()]     # noqa: E731
        try:
            g_kp, g_kd = art.get_gains()
            self._logger.debug(f"[{self._name}] first drive: sent kp={r4(kp)} kd={r4(kd)}, "
                               f"PhysX holds kp={r4(g_kp)} kd={r4(g_kd)}")
            applied = art.get_applied_actions()
            self._logger.debug(f"[{self._name}] first drive: position targets sent={r4(self._to_dof(pos_t))} "
                               f"PhysX={r4(applied.joint_positions)}")
            self._logger.debug(f"[{self._name}] first drive: max efforts {r4(art.get_max_efforts())}, "
                               f"DOF types {[str(t) for t in art.get_dof_types()]}")
        except Exception as e:
            self._logger.debug(f"[{self._name}] first drive readback failed: {type(e).__name__}: {e}")
