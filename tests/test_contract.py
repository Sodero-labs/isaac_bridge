"""The wire contract with the controller's actuator gateway (`ZmqSimTransport` on the C++ side).

1. The protocol codec: struct sizes, field offsets, teleport detection, pack and unpack.
2. The ZMQ envelope round trip: a stand-in gateway (PUB command / SUB state, as ZmqSimTransport does) talks to the
   real MessageHandler, whose commands are answered by an echo that uses parse_command / pack_state.
"""
import struct
import time

import numpy as np

from scripts import protocol

N = protocol.SIM_MAX_JOINTS
_CMD = "<qQi{n}B{p}x{d}d".format(n=N, p=(8 - (20 + N) % 8) % 8, d=5 * N)
_STATE = "<qQi4x{d}d".format(d=3 * N)


def _pad(a, fill=0.0):
    out = [fill] * N
    out[:len(a)] = list(a)
    return out


def make_command(seq, q_d, mode=protocol.MODE_POSITION, tau_d=None, ts=12345):
    """A SimCommand as SimInterface::write builds it on the C++ side."""
    n = len(q_d)
    modes = [mode] * n + [0] * (N - n)
    tau = tau_d if tau_d is not None else [0.0] * n
    return struct.pack(_CMD, ts, seq, n, *modes, *_pad(q_d), *_pad([0.0] * n), *_pad(tau), *_pad([]), *_pad([]))


def test_codec():
    assert protocol.SIM_COMMAND_SIZE == 2648
    assert protocol.SIM_STATE_SIZE == 1560

    cmd = protocol.parse_command(make_command(7, [0.5, -0.5], tau_d=[1.0, 2.0]))
    assert cmd.seq == 7 and cmd.num_joints == 2
    assert np.allclose(cmd.q_d[:2], [0.5, -0.5])
    assert np.allclose(cmd.tau_d[:2], [1.0, 2.0])
    assert not cmd.is_init

    # teleport: mode 0xFF on every active joint
    init = struct.pack(_CMD, 0, 1, 2, *([0xFF, 0xFF] + [0] * (N - 2)),
                       *_pad([1.1, 2.2]), *_pad([]), *_pad([]), *_pad([]), *_pad([]))
    icmd = protocol.parse_command(init)
    assert icmd.is_init and np.allclose(icmd.q_d[:2], [1.1, 2.2])

    # state pack -> unpack as ZmqSimTransport::recv_state does (after the '||')
    blob = protocol.pack_state(999, 3, 2, [0.5, -0.5], [0.1, -0.1], [9.0, 8.0])
    st = struct.unpack(_STATE, blob)
    assert st[0] == 999 and st[1] == 3 and st[2] == 2
    assert np.allclose(st[3:5], [0.5, -0.5])                  # q
    assert np.allclose(st[3 + N:3 + N + 2], [0.1, -0.1])      # dq


def test_zmq_roundtrip():
    import zmq
    from scripts.moderator.interface import MessageHandler

    name = "robot_arm"
    subp, pubp = 63333, 63334
    mh = MessageHandler(host="127.0.0.1", subscribe_port=subp, publish_port=pubp)

    def echo(frames):
        # [b'TM/', name, b'sim_command', <SimCommand>], multipart as the C++ side sends it
        assert frames[0] == b"TM/", frames[0]
        if frames[2] != b"sim_command":
            return
        c = protocol.parse_command(frames[3])
        n = c.num_joints
        # an ideal plant: measured q = commanded q_d
        mh.send(frames[1].decode(), [("sim_state", protocol.pack_state(c.timestamp_ns, c.seq, n, c.q_d[:n],
                                                                       np.zeros(n), c.tau_d[:n]))])

    assert mh.start_server()
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)            # gateway PUB -> bridge SUB
    sub = ctx.socket(zmq.SUB)            # bridge PUB -> gateway SUB
    try:
        pub.set(zmq.SNDHWM, 5)
        pub.connect(f"tcp://127.0.0.1:{subp}")
        sub.set(zmq.CONFLATE, 1)
        sub.set(zmq.RCVTIMEO, 5000)
        sub.setsockopt_string(zmq.SUBSCRIBE, f"TM/{name}/sim_state")
        sub.connect(f"tcp://127.0.0.1:{pubp}")
        time.sleep(0.4)                  # PUB/SUB slow-joiner settle (the C++ side waits too)

        for seq, target in enumerate([[0.5, -0.5], [1.0, -1.0], [-0.25, 0.75]], start=1):
            pub.send_multipart([b"TM/", name.encode(), b"sim_command", make_command(seq, target, tau_d=[0.3, 0.4])])
            deadline = time.monotonic() + 5.0
            while not mh.drain_inbound(echo) and time.monotonic() < deadline:     # the physics step's drain
                time.sleep(0.002)
            raw = sub.recv()
            st = struct.unpack(_STATE, raw[raw.find(b"||") + 2:])
            assert np.allclose(st[3:5], target) and st[1] == seq and st[2] == 2
    finally:
        mh.stop_server()
        pub.close(linger=0)
        sub.close(linger=0)
        ctx.term()
