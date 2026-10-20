"""Wire protocol between Isaac Sim (this bridge) and a Sodero robot controller.

This module is the Python reference of the simulator wire and MUST stay byte-compatible with the
controller side:

  actuator gateway   hardware/sim_types.h            SimCommand / SimState
                     hardware/zmq_sim_transport.cpp  ZMQ envelope + size static_asserts
  IMU gateway        imu_gateway/simulation_wire.h   ImuState
  camera gateway     rc_camera/frame_protocol.hpp    FrameHdr (camera frames)

The Sodero cloud simulator produces the same bytes; a controller cannot tell the two apart.

Envelope (PUB/SUB, routed by the robot name = the articulation prim name in the stage):

  port 53333  controller PUB -> bridge SUB, multipart  [b'TM/', name, kind, payload]
      sim_command       <SimCommand>            actuator gateway (every control cycle)
      object_teleport   <JSON>                  move a tracked object (rc:trackedObject)
  port 53334  bridge PUB -> controller SUB, single frame  b'TM/<name>/<kind>' + b'||' + payload
      sim_state         <SimState>              every physics step, every articulation
      imu_state         <ImuState>              every physics step, articulations with rc:imuLink
      contact_forces    <ContactForces>         every physics step, articulations with rc:contactForceLinks
      root_pose         <JSON>                  <= 20 Hz of sim time, every articulation (device/root_pose.py)
      object_state      <JSON>                  50 Hz (dynamic) / 2 Hz (static), tracked objects (device/rigid_object.py)
  port 53335  bridge PUB -> camera gateway SUB, single frame
      camera_pair       [FrameHdr depth][depth16][FrameHdr rgb][rgb8]   per capture, every camera
      camera_info       <JSON>                  1 Hz, intrinsics (informational)

Commands are multipart and everything published is one frame split on b'||': the asymmetry is the
C++ transport's. The state port is always the command port + 1.

Struct layout: little-endian, the C++ compiler's natural alignment reproduced with explicit pad bytes
(sizes asserted below and by the C++ static_asserts):

  SimCommand (2648 B):
    i64 timestamp_ns | u64 seq | i32 num_joints | u8 mode[64] | pad[4]
    | f64 q_d[64] | f64 dq_d[64] | f64 tau_d[64] | f64 kp[64] | f64 kd[64]

  SimState (1560 B):
    i64 sim_time_ns | u64 seq | i32 num_joints | pad[4]
    | f64 q[64] | f64 dq[64] | f64 tau[64]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# Must equal rc::SIM_MAX_JOINTS in sim_types.h
SIM_MAX_JOINTS = 64

# Drive modes — mirror the cast rc::DriveMode used in SimInterface::write.
MODE_POSITION = 0
MODE_VELOCITY = 1
MODE_TORQUE = 2
MODE_IMPEDANCE = 3
MODE_INIT = 0xFF  # set_initial_state() teleport request for that DOF (q_d carries the pose)
MODE_IDLE = 0xFE  # "not my DOF": a gateway with sim.transport.dof_offset > 0 marks the slots below its
                  # window IDLE, so several gateways can share one articulation (arm [0..3] + gripper
                  # [4..5]); the bridge merges the commands per DOF

# '<' little-endian, no implicit alignment. The 'x' runs reproduce the padding
# the C++ struct gets from natural alignment of the following f64 block.
# pad after mode[] so the f64 block is 8-byte aligned: header 20 B + n mode bytes
_CMD_PAD = (8 - (20 + SIM_MAX_JOINTS) % 8) % 8
_CMD_FMT = "<qQi{n}B{p}x{d}d".format(n=SIM_MAX_JOINTS, p=_CMD_PAD, d=5 * SIM_MAX_JOINTS)
_STATE_FMT = "<qQi4x{d}d".format(d=3 * SIM_MAX_JOINTS)

_CMD = struct.Struct(_CMD_FMT)
_STATE = struct.Struct(_STATE_FMT)

# The same layouts as numpy structured dtypes (explicit little-endian fields + void pads, so no implicit
# alignment): the per-step hot paths decode a command with one np.frombuffer (zero-copy views instead of
# unpacking hundreds of Python objects) and encode a state by filling a preallocated record. struct.Struct
# stays the reference implementation (pack_state; the tests compare the two byte for byte).
def _cmd_fields():
    fields = [("timestamp_ns", "<i8"), ("seq", "<u8"), ("num_joints", "<i4"), ("mode", "u1", (SIM_MAX_JOINTS,))]
    if _CMD_PAD:
        fields.append(("_pad", "V{}".format(_CMD_PAD)))
    fields += [(name, "<f8", (SIM_MAX_JOINTS,)) for name in ("q_d", "dq_d", "tau_d", "kp", "kd")]
    return fields


_CMD_DTYPE = np.dtype(_cmd_fields())
_STATE_DTYPE = np.dtype([
    ("sim_time_ns", "<i8"), ("seq", "<u8"), ("num_joints", "<i4"), ("_pad", "V4"),
    ("q", "<f8", (SIM_MAX_JOINTS,)), ("dq", "<f8", (SIM_MAX_JOINTS,)), ("tau", "<f8", (SIM_MAX_JOINTS,)),
])

SIM_COMMAND_SIZE = _CMD.size
SIM_STATE_SIZE = _STATE.size

# Fail loudly on import if the layout ever drifts from the C++ side.
assert SIM_COMMAND_SIZE == 2648, f"SimCommand size {SIM_COMMAND_SIZE} != 2648 — sim_types.h changed?"
assert SIM_STATE_SIZE == 1560, f"SimState size {SIM_STATE_SIZE} != 1560 — sim_types.h changed?"
assert _CMD_DTYPE.itemsize == SIM_COMMAND_SIZE, f"SimCommand dtype {_CMD_DTYPE.itemsize} != {SIM_COMMAND_SIZE}"
assert _STATE_DTYPE.itemsize == SIM_STATE_SIZE, f"SimState dtype {_STATE_DTYPE.itemsize} != {SIM_STATE_SIZE}"


@dataclass
class SimCommand:
    """Decoded actuator gateway -> bridge command. Arrays have SIM_MAX_JOINTS entries; only the first
    ``num_joints`` are meaningful."""
    timestamp_ns: int
    seq: int
    num_joints: int
    mode: np.ndarray   # uint8
    q_d: np.ndarray    # float64  target position [rad]
    dq_d: np.ndarray   # float64  target velocity [rad/s]
    tau_d: np.ndarray  # float64  target / feed-forward torque [Nm]
    kp: np.ndarray     # float64  stiffness (IMPEDANCE only) [Nm/rad]
    kd: np.ndarray     # float64  damping   (IMPEDANCE only) [Nm*s/rad]

    @property
    def is_init(self) -> bool:
        """True when this is a teleport request (SimInterface::set_initial_state marks its window's modes
        with 0xFF and carries the pose in q_d; slots below a dof_offset window are MODE_IDLE)."""
        n = self.num_joints
        if n <= 0:
            return False
        m = self.mode[:n]
        active = m != MODE_IDLE
        return bool(np.any(m == MODE_INIT)) and bool(np.all(m[active] == MODE_INIT))


def parse_command(buf: bytes) -> SimCommand:
    """Decode a SIM_COMMAND_SIZE-byte SimCommand payload. Raises ValueError if too short.

    Zero-copy: the arrays are read-only views into ``buf`` (this runs once per gateway command, about
    a thousand times a second with a few robots). Callers that need to mutate a field copy it.
    """
    if len(buf) < SIM_COMMAND_SIZE:
        raise ValueError(f"SimCommand payload {len(buf)} < {SIM_COMMAND_SIZE}")
    rec = np.frombuffer(buf, dtype=_CMD_DTYPE, count=1)[0]
    return SimCommand(
        timestamp_ns=int(rec["timestamp_ns"]),
        seq=int(rec["seq"]),
        num_joints=int(rec["num_joints"]),
        mode=rec["mode"],
        q_d=rec["q_d"],
        dq_d=rec["dq_d"],
        tau_d=rec["tau_d"],
        kp=rec["kp"],
        kd=rec["kd"],
    )


def pack_state(sim_time_ns: int, seq: int, num_joints: int,
               q, dq, tau) -> bytes:
    """Encode a SIM_STATE_SIZE-byte SimState payload. q/dq/tau are zero-padded/truncated to
    SIM_MAX_JOINTS; ``num_joints`` tells the actuator gateway how many are valid."""
    def fit(a) -> np.ndarray:
        out = np.zeros(SIM_MAX_JOINTS, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64).ravel()
        m = min(a.size, SIM_MAX_JOINTS)
        out[:m] = a[:m]
        return out

    qv, dqv, tauv = fit(q), fit(dq), fit(tau)
    return _STATE.pack(
        int(sim_time_ns), int(seq), int(num_joints),
        *qv.tolist(), *dqv.tolist(), *tauv.tolist(),
    )


class StatePacker:
    """pack_state without the per-call allocations: one preallocated SimState record per articulation,
    refilled every physics step (zero-padded like pack_state; identical bytes)."""

    __slots__ = ("_rec", "_time", "_seq", "_count", "_q", "_dq", "_tau", "_vectors")

    def __init__(self):
        self._rec = np.zeros(1, dtype=_STATE_DTYPE)
        self._time = self._rec["sim_time_ns"]
        self._seq = self._rec["seq"]
        self._count = self._rec["num_joints"]
        self._q = self._rec["q"][0]
        self._dq = self._rec["dq"][0]
        self._tau = self._rec["tau"][0]
        self._vectors = (self._q, self._dq, self._tau)

    def pack(self, sim_time_ns: int, seq: int, num_joints: int, q, dq, tau) -> bytes:
        self._time[0] = int(sim_time_ns)
        self._seq[0] = int(seq)
        self._count[0] = int(num_joints)
        for dst, src in zip(self._vectors, (q, dq, tau)):
            # PhysX supplies one-dimensional float32/64 arrays: assigning them into the wire's float64
            # view converts without an intermediate allocation. Anything else (sequences, other
            # dimensions, dtypes, ndarray subclasses) goes through the same coercion as pack_state.
            if type(src) is np.ndarray and src.ndim == 1 and src.dtype.kind == "f" and src.dtype.itemsize in (4, 8):
                a = src
            else:
                a = np.asarray(src, dtype=np.float64).ravel()
            m = min(a.size, SIM_MAX_JOINTS)
            dst[:m] = a if a.size <= SIM_MAX_JOINTS else a[:m]
            if m < SIM_MAX_JOINTS:
                dst[m:] = 0.0
        return self._rec.tobytes()


# ---------------------------------------------------------------------------
# ImuState (bridge -> IMU gateway). Published next to sim_state for articulations whose root prim
# authors rc:imuLink. Mirrors what a body-mounted IMU and two foot switches report:
#   i64 sim_time_ns | u64 seq
#   f64 quat_xyzw[4]   IMU link orientation in the world (x, y, z, w)
#   f64 gyro[3]        angular velocity, IMU frame [rad/s]
#   f64 accel[3]       proper acceleration, IMU frame [m/s^2] (gravity included:
#                      +9.81 on the up axis at rest, like a BNO055 `acceleration`)
#   f64 lin_vel_w[3]   IMU point linear velocity, world frame [m/s]
#   f64 pos_w[3]       IMU point position, world frame [m]
#   u8  contacts[2]    foot contact flags (rc:contactLinks order, 0/1), pad[6]
# = 8 + 8 + 16*8 + 2 + 6 = 152 B.
# ---------------------------------------------------------------------------
IMU_STATE_TOPIC = "imu_state"
_IMU_FMT = "<qQ16d2B6x"
_IMU = struct.Struct(_IMU_FMT)
IMU_STATE_SIZE = _IMU.size
assert IMU_STATE_SIZE == 152, f"ImuState size {IMU_STATE_SIZE} != 152"
_IMU_DTYPE = np.dtype([
    ("sim_time_ns", "<i8"), ("seq", "<u8"), ("quat_xyzw", "<f8", (4,)), ("gyro", "<f8", (3,)), ("accel", "<f8", (3,)),
    ("lin_vel_w", "<f8", (3,)), ("pos_w", "<f8", (3,)), ("contacts", "u1", (2,)), ("_pad", "V6"),
])
assert _IMU_DTYPE.itemsize == IMU_STATE_SIZE, f"ImuState dtype {_IMU_DTYPE.itemsize} != {IMU_STATE_SIZE}"


@dataclass
class ImuState:
    sim_time_ns: int
    seq: int
    quat_xyzw: np.ndarray   # (4,)
    gyro: np.ndarray        # (3,) IMU frame [rad/s]
    accel: np.ndarray       # (3,) IMU frame [m/s^2], gravity included
    lin_vel_w: np.ndarray   # (3,) world [m/s]
    pos_w: np.ndarray       # (3,) world [m]
    contacts: np.ndarray    # (2,) uint8


def pack_imu_state(sim_time_ns: int, seq: int, quat_xyzw, gyro, accel, lin_vel_w, pos_w, contacts) -> bytes:
    def v(a, n):
        out = np.zeros(n, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64).ravel()
        k = min(a.size, n)
        out[:k] = a[:k]
        return out.tolist()
    c = np.zeros(2, dtype=np.uint8)
    ca = np.asarray(contacts).ravel()
    for i in range(min(2, ca.size)):
        c[i] = 1 if bool(ca[i]) else 0
    return _IMU.pack(int(sim_time_ns), int(seq), *v(quat_xyzw, 4), *v(gyro, 3), *v(accel, 3),
                     *v(lin_vel_w, 3), *v(pos_w, 3), int(c[0]), int(c[1]))


class ImuPacker:
    """pack_imu_state on a preallocated record (identical bytes, no per-call list building)."""

    __slots__ = ("_rec", "_time", "_seq", "_vectors", "_contacts")

    def __init__(self):
        self._rec = np.zeros(1, dtype=_IMU_DTYPE)
        self._time = self._rec["sim_time_ns"]
        self._seq = self._rec["seq"]
        self._vectors = tuple(self._rec[name][0] for name in
                              ("quat_xyzw", "gyro", "accel", "lin_vel_w", "pos_w"))
        self._contacts = self._rec["contacts"][0]

    def pack(self, sim_time_ns: int, seq: int, quat_xyzw, gyro, accel, lin_vel_w, pos_w, contacts) -> bytes:
        self._time[0] = int(sim_time_ns)
        self._seq[0] = int(seq)
        for dst, src in zip(self._vectors, (quat_xyzw, gyro, accel, lin_vel_w, pos_w)):
            if type(src) is np.ndarray and src.ndim == 1 and src.dtype.kind == "f" and src.dtype.itemsize in (4, 8):
                a = src
            else:
                a = np.asarray(src, dtype=np.float64).ravel()
            n = dst.size
            k = min(a.size, n)
            dst[:k] = a if a.size <= n else a[:k]
            if k < n:
                dst[k:] = 0.0
        c = self._contacts
        c[:] = 0
        ca = np.asarray(contacts).ravel()
        for i in range(min(2, ca.size)):
            c[i] = 1 if bool(ca[i]) else 0
        return self._rec.tobytes()


# ---------------------------------------------------------------------------
# ContactForces — measured contact loads of a robot's rc:contactForceLinks (same PUB, own topic)
#   i64 sim_time_ns    simulation time of the physics step the forces belong to
#   u64 seq            per-robot counter
#   u8  count          links carried (<= 8), in rc:contactForceLinks order; pad[7]
#   f32 normal_z[8]    world-z component of each link's net contact force [N] (unused entries 0)
# = 8 + 8 + 1 + 7 + 32 = 56 B. A controller can gate support changes (lift a foot, release an arm) on how
# much weight each link carries; imu_state's two geometric foot flags cannot say that.
# ---------------------------------------------------------------------------
CONTACT_FORCES_TOPIC = "contact_forces"
CONTACT_FORCES_MAX = 8
_CONTACT_FORCES = struct.Struct("<qQB7x8f")
CONTACT_FORCES_SIZE = _CONTACT_FORCES.size
assert CONTACT_FORCES_SIZE == 56, f"ContactForces size {CONTACT_FORCES_SIZE} != 56"


def pack_contact_forces(sim_time_ns: int, seq: int, normal_z) -> bytes:
    values = np.zeros(CONTACT_FORCES_MAX, dtype=np.float32)
    forces = np.asarray(normal_z, dtype=np.float32).ravel()
    count = min(forces.size, CONTACT_FORCES_MAX)
    values[:count] = forces[:count]
    return _CONTACT_FORCES.pack(int(sim_time_ns), int(seq), count, *values.tolist())


def parse_contact_forces(buf: bytes):
    """(sim_time_ns, seq, forces (count,) float64)."""
    if len(buf) < CONTACT_FORCES_SIZE:
        raise ValueError(f"ContactForces payload {len(buf)} < {CONTACT_FORCES_SIZE}")
    f = _CONTACT_FORCES.unpack_from(buf, 0)
    count = min(int(f[2]), CONTACT_FORCES_MAX)
    return int(f[0]), int(f[1]), np.asarray(f[3:3 + count], dtype=np.float64)


def parse_imu_state(buf: bytes) -> ImuState:
    if len(buf) < IMU_STATE_SIZE:
        raise ValueError(f"ImuState payload {len(buf)} < {IMU_STATE_SIZE}")
    f = _IMU.unpack_from(buf, 0)
    d = np.array(f[2:18], dtype=np.float64)
    return ImuState(sim_time_ns=f[0], seq=f[1], quat_xyzw=d[0:4], gyro=d[4:7], accel=d[7:10],
                    lin_vel_w=d[10:13], pos_w=d[13:16], contacts=np.array(f[18:20], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Camera frames (bridge -> camera gateway, `isaac` input).
#
# Byte-exact mirror of the camera gateway's rc_camera/frame_protocol.hpp:
#
#   FrameHdr (64 B):
#     char magic[4] "RCIM" | u32 version=1 | u32 pair_seq | u32 width | u32 height
#     | u32 step_bytes | u32 payload_bytes | u32 encoding
#     | i64 observed_at_sync_ns | f32 depth_unit_m | u32 flags | char frame_id[16]
#
# One frame per capture on the camera PUB socket (config.CAMERA_PUB_PORT), never on the joint socket:
#   b'TM/<name>/camera_pair' + b'||' + [hdr depth][depth16 mm][hdr rgb][rgb8]
#
# With FLAG_CAPTURE_UNIX_TIME the headers carry the capture time on the wall clock (time.time_ns()) and
# the camera gateway converts it to its own clock (accepted from 20 ms in its future to 2 s in its past:
# keep the two computers' clocks synchronized). Without the flag observed_at_sync_ns is 0 and the gateway
# stamps the arrival time. The bridge sets the flag unless RC_CAMERA_CAPTURE_TIME=0.
# ---------------------------------------------------------------------------
FRAME_MAGIC = b"RCIM"
FRAME_PROTOCOL_VERSION = 1
FRAME_HDR_FMT = "<4s7IqfI16s"
_FRAME_HDR = struct.Struct(FRAME_HDR_FMT)
FRAME_HDR_SIZE = _FRAME_HDR.size
assert FRAME_HDR_SIZE == 64, f"FrameHdr size {FRAME_HDR_SIZE} != 64 — frame_protocol.hpp changed?"

# rc_camera::Encoding
ENCODING_RGB8 = 1      # 3 B/px row-major, step_bytes = width*3, depth_unit_m = 0
ENCODING_DEPTH16 = 2   # uint16 millimetres, step_bytes = width*2, depth_unit_m = 0.001
ENCODING_DEPTH32F = 3  # float32 metres (not produced by the bridge)

# rc_camera::FrameFlags
FLAG_NOTIFY_ONLY = 1 << 0        # header-only notification event; pixels must be pulled
FLAG_DEPTH_FIRST = 1 << 1        # pair blob layout: depth frame then rgb frame
FLAG_CAPTURE_UNIX_TIME = 1 << 2  # observed_at_sync_ns is a wall-clock (Unix) capture time

DEPTH16_UNIT_M = 0.001
DEPTH16_MAX_MM = 0xFFFF
FRAME_ID_MAX_LEN = 15       # char[16], NUL-terminated on the C++ side

CAMERA_PAIR_TOPIC = "camera_pair"   # b'TM/<name>/camera_pair||<blob>'
CAMERA_INFO_TOPIC = "camera_info"   # b'TM/<name>/camera_info||<json>' (1 Hz, intrinsics)


@dataclass
class FrameHdr:
    """Decoded 64-byte frame header (field names == the C++ struct)."""
    magic: bytes
    version: int
    pair_seq: int
    width: int
    height: int
    step_bytes: int
    payload_bytes: int
    encoding: int
    observed_at_sync_ns: int
    depth_unit_m: float
    flags: int
    frame_id: str

    @property
    def is_valid(self) -> bool:
        return self.magic == FRAME_MAGIC and self.version == FRAME_PROTOCOL_VERSION


def _encode_frame_id(frame_id: str) -> bytes:
    raw = frame_id.encode("utf-8")
    if len(raw) > FRAME_ID_MAX_LEN:
        # A 16-byte name would leave no NUL terminator for the C++ reader; refuse rather than silently
        # truncating a frame name the transform tree keys on.
        raise ValueError(
            f"frame_id '{frame_id}' is {len(raw)} bytes; FrameHdr::frame_id holds at most "
            f"{FRAME_ID_MAX_LEN} + NUL")
    return raw.ljust(16, b"\0")


def pack_frame_hdr(*, pair_seq: int, width: int, height: int, encoding: int,
                   payload_bytes: int, frame_id: str, step_bytes: "int | None" = None,
                   observed_at_sync_ns: int = 0, depth_unit_m: "float | None" = None,
                   flags: int = 0) -> bytes:
    """Encode one FrameHdr. ``step_bytes``/``depth_unit_m`` default from ``encoding``."""
    if step_bytes is None:
        step_bytes = {ENCODING_RGB8: width * 3, ENCODING_DEPTH16: width * 2,
                      ENCODING_DEPTH32F: width * 4}.get(int(encoding))
        if step_bytes is None:
            raise ValueError(f"unknown encoding {encoding}; pass step_bytes explicitly")
    if depth_unit_m is None:
        depth_unit_m = {ENCODING_DEPTH16: DEPTH16_UNIT_M, ENCODING_DEPTH32F: 1.0}.get(int(encoding), 0.0)
    return _FRAME_HDR.pack(
        FRAME_MAGIC, FRAME_PROTOCOL_VERSION, int(pair_seq) & 0xFFFFFFFF,
        int(width), int(height), int(step_bytes), int(payload_bytes), int(encoding),
        int(observed_at_sync_ns), float(depth_unit_m), int(flags),
        _encode_frame_id(frame_id))


def unpack_frame_hdr(buf: bytes, offset: int = 0) -> FrameHdr:
    """Decode a FrameHdr at ``offset``. Raises ValueError if too short or the magic is wrong."""
    if len(buf) - offset < FRAME_HDR_SIZE:
        raise ValueError(f"FrameHdr needs {FRAME_HDR_SIZE} B, got {len(buf) - offset}")
    f = _FRAME_HDR.unpack_from(buf, offset)
    hdr = FrameHdr(
        magic=f[0], version=f[1], pair_seq=f[2], width=f[3], height=f[4], step_bytes=f[5],
        payload_bytes=f[6], encoding=f[7], observed_at_sync_ns=f[8], depth_unit_m=f[9],
        flags=f[10], frame_id=f[11].split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    if hdr.magic != FRAME_MAGIC:
        raise ValueError(f"bad FrameHdr magic {hdr.magic!r}")
    return hdr


def depth_m_to_mm(depth_m) -> np.ndarray:
    """float metres -> uint16 millimetres (DEPTH16), RealSense convention.

    * NaN / +-inf (no hit, beyond the far clip) and negative values -> 0 = invalid.
    * finite values above 65.535 m saturate at 65535 (a real but unmeasurable range;
      consumers clip at their own depth_max long before that).
    """
    d = np.asarray(depth_m, dtype=np.float32)
    mm = np.rint(d * 1000.0)
    valid = np.isfinite(mm) & (mm >= 0.0)
    mm = np.where(valid, np.minimum(mm, float(DEPTH16_MAX_MM)), 0.0)
    return mm.astype(np.uint16)


def pack_frame(*, pair_seq: int, encoding: int, pixels: np.ndarray, frame_id: str,
               observed_at_sync_ns: int = 0, flags: int = 0) -> bytes:
    """[FrameHdr][pixels] for one image. ``pixels`` is (H, W) uint16 for DEPTH16 or
    (H, W, 3) uint8 for RGB8; it is made C-contiguous, so pass a view freely."""
    px = np.ascontiguousarray(pixels)
    if encoding == ENCODING_DEPTH16:
        if px.ndim != 2 or px.dtype != np.uint16:
            raise ValueError(f"DEPTH16 expects (H, W) uint16, got {px.shape} {px.dtype}")
    elif encoding == ENCODING_RGB8:
        if px.ndim != 3 or px.shape[2] != 3 or px.dtype != np.uint8:
            raise ValueError(f"RGB8 expects (H, W, 3) uint8, got {px.shape} {px.dtype}")
    else:
        raise ValueError(f"pack_frame: unsupported encoding {encoding}")
    h, w = int(px.shape[0]), int(px.shape[1])
    hdr = pack_frame_hdr(pair_seq=pair_seq, width=w, height=h, encoding=encoding,
                         payload_bytes=px.nbytes, frame_id=frame_id,
                         observed_at_sync_ns=observed_at_sync_ns, flags=flags)
    return hdr + px.tobytes()


def pack_frame_pair(*, pair_seq: int, depth_mm: np.ndarray, rgb: np.ndarray, frame_id: str,
                    observed_at_sync_ns: int = 0, flags: int = 0) -> bytes:
    """[hdr depth][depth16][hdr rgb][rgb8], the pair blob layout (kDepthFirst) of the camera gateway.
    ``flags`` is OR-ed into both headers next to FLAG_DEPTH_FIRST (FLAG_CAPTURE_UNIX_TIME with a
    wall-clock capture time)."""
    if depth_mm.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"depth {depth_mm.shape[:2]} and rgb {rgb.shape[:2]} sizes differ")
    return (pack_frame(pair_seq=pair_seq, encoding=ENCODING_DEPTH16, pixels=depth_mm,
                       frame_id=frame_id, observed_at_sync_ns=observed_at_sync_ns,
                       flags=FLAG_DEPTH_FIRST | flags)
            + pack_frame(pair_seq=pair_seq, encoding=ENCODING_RGB8, pixels=rgb,
                         frame_id=frame_id, observed_at_sync_ns=observed_at_sync_ns,
                         flags=FLAG_DEPTH_FIRST | flags))


def split_frames(blob: bytes) -> "list[tuple[FrameHdr, bytes]]":
    """Walk a concatenation of [hdr][pixels] records (a pair blob, or a single frame) and
    return [(hdr, pixel_bytes), ...]. Raises ValueError on a truncated record."""
    out = []
    off = 0
    n = len(blob)
    while off < n:
        hdr = unpack_frame_hdr(blob, off)
        start = off + FRAME_HDR_SIZE
        end = start + hdr.payload_bytes
        if end > n:
            raise ValueError(f"frame at {off}: payload {hdr.payload_bytes} B exceeds blob ({n} B)")
        out.append((hdr, blob[start:end]))
        off = end
    return out


def frame_to_array(hdr: FrameHdr, pixels: bytes) -> np.ndarray:
    """View pixel bytes as the numpy array their header describes."""
    if hdr.encoding == ENCODING_DEPTH16:
        return np.frombuffer(pixels, dtype=np.uint16).reshape(hdr.height, hdr.width)
    if hdr.encoding == ENCODING_RGB8:
        return np.frombuffer(pixels, dtype=np.uint8).reshape(hdr.height, hdr.width, 3)
    if hdr.encoding == ENCODING_DEPTH32F:
        return np.frombuffer(pixels, dtype=np.float32).reshape(hdr.height, hdr.width)
    raise ValueError(f"unknown encoding {hdr.encoding}")


def split_zmq_frame(raw: bytes) -> "tuple[bytes, bytes]":
    """b'TM/<name>/<data_type>||<payload>' -> (topic, payload); the state/camera envelope."""
    i = raw.find(b"||")
    if i < 0:
        raise ValueError("ZMQ frame has no '||' divider")
    return raw[:i], raw[i + 2:]
