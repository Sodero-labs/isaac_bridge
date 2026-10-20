"""Gateway commands: the per-DOF merge, the non-finite guard and the drive resolution."""
from __future__ import annotations

import numpy as np
import pytest

from fakes import command, make_robot
from scripts import protocol

M = protocol.SIM_MAX_JOINTS


def _mixed_command(n, q_d, *, init_at=(), idle_at=()):
    modes = np.full(n, protocol.MODE_POSITION, dtype=np.uint8)
    modes[list(init_at)] = protocol.MODE_INIT
    modes[list(idle_at)] = protocol.MODE_IDLE
    return command(n, q_d, modes=modes, dq_d=np.arange(n) * 0.5, tau_d=np.arange(n) * 0.25,
                   kp=np.full(n, 30.0), kd=np.full(n, 1.5))


def _merge(payload):
    a, _ = make_robot()
    a._cmd_dirty = False
    a._handle_sim_command(payload)
    return {f: getattr(a, f).copy() for f in
            ("_mode", "_q_d", "_dq_d", "_tau_d", "_kp", "_kd", "_tp_mask", "_tp_q")} | {"dirty": a._cmd_dirty}


# ------------------------------------------------------------------ merge
def test_a_whole_window_command_is_merged_as_sent():
    n = 6
    got = _merge(_mixed_command(n, np.linspace(0.1, 0.6, n)))
    assert got["dirty"] is True
    assert np.array_equal(got["_mode"][:n], np.full(n, protocol.MODE_POSITION))
    assert np.allclose(got["_q_d"][:n], np.linspace(0.1, 0.6, n))
    assert np.allclose(got["_dq_d"][:n], np.arange(n) * 0.5)
    assert np.allclose(got["_tau_d"][:n], np.arange(n) * 0.25)
    assert np.allclose(got["_kp"][:n], 30.0) and np.allclose(got["_kd"][:n], 1.5)
    assert not got["_tp_mask"].any()
    assert np.array_equal(got["_mode"][n:], np.full(M - n, protocol.MODE_IDLE))


def test_teleport_and_idle_slots_are_merged_per_dof():
    n = 6
    teleport = _merge(_mixed_command(n, np.linspace(0.1, 0.6, n), init_at=(2,)))
    assert list(np.flatnonzero(teleport["_tp_mask"])) == [2]
    assert np.isclose(teleport["_tp_q"][2], 0.3)
    assert teleport["_mode"][2] == protocol.MODE_IDLE          # INIT is a teleport, not a drive mode
    assert teleport["_mode"][3] == protocol.MODE_POSITION

    hole = _merge(_mixed_command(n, np.linspace(0.1, 0.6, n), idle_at=(4,)))
    assert hole["_mode"][4] == protocol.MODE_IDLE              # an unowned DOF keeps its idle default
    assert np.isclose(hole["_kp"][4], 0.0)                     # ... and is not given this command's gains
    assert np.isclose(hole["_kp"][3], 30.0)


def test_two_gateways_share_one_robot_through_disjoint_windows():
    a, _ = make_robot(6)
    a.update_target("sim_command", command(4, [0.1, 0.2, 0.3, 0.4]))                         # arm: DOF 0..3
    a.update_target("sim_command", command(6, [9, 9, 9, 9, 0.5, 0.6],
                                           modes=[protocol.MODE_IDLE] * 4 + [protocol.MODE_POSITION] * 2))  # gripper
    np.testing.assert_allclose(a._q_d[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert a._mode[:6].tolist() == [protocol.MODE_POSITION] * 6


# ------------------------------------------------------------------ non-finite guard
def test_non_finite_command_values_never_reach_the_drive(caplog):
    a, pv = make_robot(3)
    nan = float("nan")
    # a gateway's limp frame after it read a NaN state: IMPEDANCE, q_d NaN, kp 0, kd 5000
    a._cmd_dirty = False
    a.update_target("sim_command", command(3, [nan, nan, nan], mode=protocol.MODE_IMPEDANCE, kp=[0, 0, 0],
                                           kd=[5000, 5000, 5000]))
    assert a._mode[:3].tolist() == [protocol.MODE_IDLE] * 3                # nothing owned, nothing written
    assert np.isfinite(a._q_d).all() and not a._cmd_dirty
    assert "non-finite q_d" in caplog.text
    # a partly bad command: the finite DOFs are applied, the NaN one keeps its previous command
    a.update_target("sim_command", command(3, [0.1, 0.2, 0.3]))
    a.update_target("sim_command", command(3, [0.5, nan, 0.7], kd=[0, float("inf"), 0]))
    np.testing.assert_allclose(a._q_d[:3], [0.5, 0.2, 0.7])
    assert a._mode[:3].tolist() == [protocol.MODE_POSITION] * 3
    a.synchronize()
    assert np.isfinite(pv.last("pos")).all()
    # an INIT teleport with a NaN pose is ignored for that DOF
    a.update_target("sim_command", command(3, [nan, 1.0, nan], mode=protocol.MODE_INIT))
    assert a._tp_mask[:3].tolist() == [False, True, False] and np.isfinite(a._tp_q).all()


def test_finite_commands_take_the_fast_path_untouched():
    a, _ = make_robot(3)
    a.update_target("sim_command", command(3, [0.1, 0.2, 0.3], kp=[1, 2, 3], kd=[4, 5, 6]))
    np.testing.assert_allclose(a._q_d[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(a._kp[:3], [1, 2, 3])
    assert a._cmd_dirty and a._nonfinite_cmds == 0


# ------------------------------------------------------------------ drive resolution
@pytest.mark.parametrize("n", [1, 12, 18, 24])
@pytest.mark.parametrize("authored,feedforward", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("latched", [False, True])
def test_the_all_position_drive_matches_the_general_resolver(n, authored, feedforward, latched):
    """Byte for byte, and with arrays of its own (the servo models adjust them in place)."""
    rng = np.random.default_rng(71)
    a, _ = make_robot(n)
    a._latch_enabled = latched
    arrays = [np.full(M, protocol.MODE_POSITION, dtype=np.uint8)] + [rng.normal(size=M).astype(np.float32)
                                                                     for _ in range(5)]
    a._mode, a._q_d, a._dq_d, a._tau_d, a._kp, a._kd = arrays
    a._latched = tuple(x.copy() for x in arrays) if latched else None
    a._usd_kp = rng.uniform(1, 100, n).astype(np.float32) if authored else None
    a._usd_kd = rng.uniform(.1, 10, n).astype(np.float32) if authored else None
    a._usd_vel_ff = feedforward
    q = rng.normal(size=n)
    general = a._resolve_mixed(n, q, *(x[:n] for x in a._command_in_force()))
    resolved = a._resolve_drive(q)
    for got, want in zip(resolved, general):
        assert got.dtype == np.float32
        assert got.tobytes() == want.tobytes()
        for source in arrays + list(a._latched or ()) + [a._usd_kp, a._usd_kd, q]:
            if source is not None:
                assert not np.shares_memory(got, source)


def test_mixed_modes_get_their_own_gains_and_targets():
    a, _ = make_robot(5)
    modes = [protocol.MODE_IDLE, protocol.MODE_TORQUE, protocol.MODE_POSITION, protocol.MODE_IMPEDANCE,
             protocol.MODE_VELOCITY]
    a.update_target("sim_command", command(5, [0.1, 0.2, 0.3, 0.4, 0.5], modes=modes, dq_d=[1, 2, 3, 4, 5],
                                           tau_d=[6, 7, 8, 9, 10], kp=[0, 0, 0, 40, 0], kd=[0, 0, 0, 2, 0]))
    q_now = np.full(5, -1.0)
    kp, kd, pos_t, vel_t, eff = a._resolve_drive(q_now)
    np.testing.assert_allclose(kp, [0, 0, 1e7, 40, 0])
    np.testing.assert_allclose(kd, [1.0, 0, 1e6, 2, 1e4])       # IDLE: light damping; TORQUE: none
    np.testing.assert_allclose(pos_t, [-1, -1, 0.3, 0.4, -1], rtol=1e-6)
    np.testing.assert_allclose(vel_t, [0, 0, 3, 4, 5])
    np.testing.assert_allclose(eff, [0, 7, 8, 9, 10])
