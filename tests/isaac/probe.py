"""Talk to a running bridge the way a controller's gateways do, and check what comes back.

    python tests/isaac/probe.py --robot turtlebot3_burger [--host 127.0.0.1] [--seconds 10]
        [--drive velocity --target 4.0,4.0] [--camera cam0] [--expect-move 0.1] [--json]

* subscribes to the state port (every TM/<robot>/... frame) and, with --camera, to the camera port;
* with --drive, sends sim_command frames at 100 Hz as the actuator gateway does (all joints in the given mode;
  velocity targets in rad/s, position targets in rad), then limp frames (IMPEDANCE, kp 0), as a gateway
  switching its drive off;
* prints per-topic rates and a summary; the exit code says whether the expectations held.
Needs numpy and pyzmq only.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import protocol  # noqa: E402

MODES = {"position": protocol.MODE_POSITION, "velocity": protocol.MODE_VELOCITY,
         "torque": protocol.MODE_TORQUE, "impedance": protocol.MODE_IMPEDANCE}


def command(seq: int, n: int, mode: int, target, kd: float = 0.0) -> bytes:
    """A SimCommand as the actuator gateway builds it (kp = kd = 0 except in the limp frame)."""
    N = protocol.SIM_MAX_JOINTS
    modes = [mode] * n + [0] * (N - n)
    q = np.zeros(N); dq = np.zeros(N); tau = np.zeros(N); kp = np.zeros(N); kdv = np.zeros(N)
    t = np.zeros(n) if target is None else np.resize(np.asarray(target, dtype=np.float64), n)
    if mode == protocol.MODE_POSITION:
        q[:n] = t
    elif mode == protocol.MODE_VELOCITY:
        dq[:n] = t
    elif mode == protocol.MODE_TORQUE:
        tau[:n] = t
    elif mode == protocol.MODE_IMPEDANCE:
        q[:n] = t
        kdv[:n] = kd
    fmt = "<qQi{n}B{p}x{d}d".format(n=N, p=(8 - (20 + N) % 8) % 8, d=5 * N)
    return struct.pack(fmt, 0, seq, n, *modes, *q, *dq, *tau, *kp, *kdv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="robot name (the articulation prim name)")
    ap.add_argument("--host", default="127.0.0.1", help="computer running Isaac Sim")
    ap.add_argument("--port", type=int, default=53333, help="command port; state = +1")
    ap.add_argument("--camera-port", type=int, default=53335)
    ap.add_argument("--camera", default="", help="camera prim name to subscribe to")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to listen (and drive)")
    ap.add_argument("--drive", choices=sorted(MODES), default="", help="drive every joint in this mode")
    ap.add_argument("--target", default="", help="comma separated targets, repeated over the joints")
    ap.add_argument("--hold-first", action="store_true", help="position mode: hold the first reported pose")
    ap.add_argument("--expect-move", type=float, default=0.0, help="root must move at least this far [m]")
    ap.add_argument("--json", action="store_true", help="one-line JSON summary")
    a = ap.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, f"TM/{a.robot}/".encode())
    sub.connect(f"tcp://{a.host}:{a.port + 1}")
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://{a.host}:{a.port}")
    cam = None
    if a.camera:
        cam = ctx.socket(zmq.SUB)
        cam.setsockopt(zmq.SUBSCRIBE, f"TM/{a.camera}/".encode())
        cam.connect(f"tcp://{a.host}:{a.camera_port}")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    if cam is not None:
        poller.register(cam, zmq.POLLIN)

    counts: "dict[str, int]" = {}
    first = {}
    last = {}
    cam_info = None
    cam_stats = {"pairs": 0, "bad": 0, "max_age_ms": None, "min_age_ms": None}
    target = [float(x) for x in a.target.split(",")] if a.target else None
    mode = MODES.get(a.drive)
    n_joints = None
    hold = None
    seq = 0
    t0 = time.monotonic()
    next_cmd = t0
    driving = mode is not None
    while time.monotonic() - t0 < a.seconds:
        for sock, _ in poller.poll(5):
            raw = sock.recv()
            topic, payload = protocol.split_zmq_frame(raw)
            kind = topic.decode().split("/")[-1]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "sim_state":
                rec = struct.unpack_from("<qQi", payload)
                n = rec[2]
                q = np.frombuffer(payload, dtype="<f8", count=n, offset=24)
                dq = np.frombuffer(payload, dtype="<f8", count=n, offset=24 + 8 * protocol.SIM_MAX_JOINTS)
                last[kind] = {"t": rec[0] / 1e9, "seq": rec[1], "n": n, "q": q.round(4).tolist(),
                              "dq": dq.round(3).tolist()}
                n_joints = n
                if hold is None:
                    hold = q.copy()
            elif kind == "imu_state":
                s = protocol.parse_imu_state(payload)
                last[kind] = {"t": s.sim_time_ns / 1e9, "accel": s.accel.round(3).tolist(),
                              "gyro": s.gyro.round(3).tolist(), "quat_xyzw": s.quat_xyzw.round(4).tolist(),
                              "contacts": s.contacts.tolist()}
            elif kind in ("root_pose", "object_state", "camera_info"):
                doc = json.loads(payload)
                last[kind] = doc
                if kind == "camera_info":
                    cam_info = doc
            elif kind == "contact_forces":
                t, s_, f = protocol.parse_contact_forces(payload)
                last[kind] = {"t": t / 1e9, "forces": f.round(2).tolist()}
            elif kind == "camera_pair":
                try:
                    (dh, dpx), (rh, rpx) = protocol.split_frames(payload)
                    depth = protocol.frame_to_array(dh, dpx)
                    rgb = protocol.frame_to_array(rh, rpx)
                    cam_stats["pairs"] += 1
                    if dh.flags & protocol.FLAG_CAPTURE_UNIX_TIME:
                        age = (time.time_ns() - dh.observed_at_sync_ns) / 1e6
                        cam_stats["max_age_ms"] = max(cam_stats["max_age_ms"] or age, age)
                        cam_stats["min_age_ms"] = min(cam_stats["min_age_ms"] or age, age)
                    last[kind] = {"seq": dh.pair_seq, "size": [dh.width, dh.height], "frame_id": dh.frame_id,
                                  "flags": dh.flags, "depth_valid": round(float(np.count_nonzero(depth)) / depth.size, 3),
                                  "depth_centre_m": round(float(depth[dh.height // 2, dh.width // 2]) / 1000.0, 3),
                                  "rgb_mean": [round(float(c), 1) for c in rgb.reshape(-1, 3).mean(axis=0)]}
                except Exception as e:  # noqa: BLE001
                    cam_stats["bad"] += 1
                    last[kind] = {"error": str(e)}
            if kind not in first and kind in last:
                first[kind] = last[kind]
        now = time.monotonic()
        if driving and n_joints and now >= next_cmd:
            next_cmd = now + 0.01
            seq += 1
            tgt = hold if (a.hold_first and mode == protocol.MODE_POSITION) else target
            pub.send_multipart([b"TM/", a.robot.encode(), b"sim_command", command(seq, n_joints, mode, tgt)])
    if mode is not None and n_joints:
        for _ in range(5):                                   # switch the drive off: limp, like a gateway
            seq += 1
            pub.send_multipart([b"TM/", a.robot.encode(), b"sim_command",
                                command(seq, n_joints, protocol.MODE_IMPEDANCE, last["sim_state"]["q"], kd=5000.0)])
            time.sleep(0.01)
    wall = time.monotonic() - t0
    rates = {k: round(v / wall, 1) for k, v in sorted(counts.items())}
    moved = None
    if "root_pose" in first and "root_pose" in last:
        moved = float(np.linalg.norm(np.subtract(last["root_pose"]["pos"], first["root_pose"]["pos"])))
    summary = {"rates_hz": rates, "first": first, "last": last, "camera": cam_stats, "camera_info": cam_info,
               "root_moved_m": None if moved is None else round(moved, 3), "commands_sent": seq}
    ok = "sim_state" in counts
    if a.camera:
        ok = ok and cam_stats["pairs"] > 0 and cam_stats["bad"] == 0
    if a.expect_move:
        ok = ok and moved is not None and moved >= a.expect_move
    summary["ok"] = ok
    print(json.dumps(summary, indent=None if a.json else 2))
    for s in (sub, pub, cam):
        if s is not None:
            s.close(linger=0)
    ctx.term()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
