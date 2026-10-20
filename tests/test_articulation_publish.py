"""What a robot publishes: the sim_state packing and the effort rules (drive torque where the drive has a force
limit, the measured reaction torque elsewhere, and a fallback that says what it publishes)."""
from __future__ import annotations

import logging

import numpy as np

from fakes import command, make_robot
from scripts import protocol


def _robot(n=3, *, kp=None, am=None, fmax=30.0):
    a, _ = make_robot(n)
    a._fmax_canon = False if fmax is None else np.full(n, float(fmax))
    a._last_drive = None if kp is None else (
        np.asarray(kp, dtype=np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32),
        np.zeros(n, np.float32), np.zeros(n, np.float32))
    a._am = am
    a._am_tau = None if am is None else np.zeros(n)
    return a


# ------------------------------------------------------------------ when the measured efforts are read
def test_measured_effort_skipped_when_every_dof_is_pd_driven():
    assert _robot(kp=[35.0, 35.0, 100.0])._measured_effort_needed(3) is False


def test_measured_effort_kept_for_a_kp_zero_dof():
    assert _robot(kp=[35.0, 0.0, 100.0])._measured_effort_needed(3) is True     # IDLE / VELOCITY / TORQUE DOF


def test_measured_effort_kept_before_the_first_drive():
    assert _robot(kp=None)._measured_effort_needed(3) is True


def test_measured_effort_skipped_when_servo_models_cover_the_kp_zero_dofs():
    am = type("AM", (), {"mask": np.array([False, True, False])})()
    a = _robot(kp=[35.0, 0.0, 35.0], am=am)      # the model drives DOF 1 (kp 0) and overrides its tau
    assert a._measured_effort_needed(3) is False


def test_measured_effort_kept_when_the_drive_has_no_force_limit():
    # no USD maxForce and no PhysX limit: the drive estimate is not published, the reaction torque must be read
    assert _robot(kp=[35.0, 35.0, 100.0], fmax=None)._measured_effort_needed(3) is True
    a = _robot(kp=[35.0, 35.0, 100.0])
    a._fmax_canon = None                         # not looked up yet
    assert a._measured_effort_needed(3) is True


def test_measured_effort_kept_on_a_dof_count_mismatch():
    assert _robot(kp=[35.0, 35.0])._measured_effort_needed(3) is True


# ------------------------------------------------------------------ the published frames
def test_sim_state_frame_is_the_protocol_packing():
    a, _ = make_robot(2, sim_time=1.5)
    a._actual = (np.ones(2), np.zeros(2), np.zeros(2))
    frames = dict(a.get_actual())
    assert frames["sim_state"] == protocol.pack_state(int(1.5e9), 1, 2, np.ones(2), np.zeros(2), np.zeros(2))


def test_nothing_is_published_before_the_first_state():
    a, _ = make_robot()
    assert a.get_actual() == []


def test_drive_torque_estimate_needs_a_force_limit_else_the_reaction_torque_is_published():
    # An authored stiffness of 1e5 per degree without a maxForce: a fraction of a degree of hold error would read as
    # kilonewton-metres. Without a usable force limit the published effort is the measured reaction torque.
    a, _ = make_robot(3)
    a._usd_kp = np.full(3, 5.7e6, dtype=np.float32)
    a._usd_kd = np.full(3, 5.7e5, dtype=np.float32)
    a._usd_vel_ff = False
    a._art.get_measured_joint_efforts = lambda: np.full((1, 3), 3.0, dtype=np.float32)
    a.update_target("sim_command", command(3, [0.5, 0.5, 0.5]))
    a.synchronize()
    assert a._fmax_canon is False                                            # neither USD nor PhysX limit
    np.testing.assert_allclose(a._actual[2], 3.0)                            # reaction torque, not ~2e6
    # with a limit the estimate is published, clamped to it
    b, _ = make_robot(3)
    b._usd_kp = np.full(3, 5.7e6, dtype=np.float32)
    b._usd_kd = np.full(3, 5.7e5, dtype=np.float32)
    b._usd_vel_ff = False
    b._fmax_canon = None                                                     # look it up
    b._art.get_max_efforts = lambda: np.full((1, 3), 30.0, dtype=np.float32)
    b._art.get_measured_joint_efforts = lambda: np.full((1, 3), 3.0, dtype=np.float32)
    b.update_target("sim_command", command(3, [0.5, 0.5, 0.5]))
    b.synchronize()
    np.testing.assert_allclose(b._fmax_canon, 30.0)
    np.testing.assert_allclose(b._actual[2], 30.0)


def test_physx_flt_max_is_not_a_force_limit():
    # An unauthored maxForce reads back from PhysX as 3.4e38: finite, but "unlimited".
    a, _ = make_robot(3)
    a._usd_kp = np.full(3, 5.7e6, dtype=np.float32)
    a._usd_kd = np.full(3, 5.7e5, dtype=np.float32)
    a._usd_vel_ff = False
    a._fmax_canon = None
    a._art.get_max_efforts = lambda: np.full((1, 3), 3.4028234663852886e+38, dtype=np.float32)
    a._art.get_measured_joint_efforts = lambda: np.full((1, 3), 3.0, dtype=np.float32)
    a.update_target("sim_command", command(3, [0.5, 0.5, 0.5]))
    a.synchronize()
    assert a._fmax_canon is False
    np.testing.assert_allclose(a._actual[2], 3.0)


# ------------------------------------------------------------------ the drive-effort fallback
class _EffortArt:
    def __init__(self, tau=None, fail=False):
        self.calls = 0
        self._tau = tau
        self._fail = fail

    def get_measured_joint_efforts(self):
        self.calls += 1
        if self._fail:
            raise RuntimeError("tensor view gone")
        return np.asarray([self._tau], dtype=np.float32)


def test_effort_fallback_reads_measured_lazily_when_the_read_was_skipped(caplog):
    a = _robot(kp=[35.0] * 3)
    art = _EffortArt(tau=[1.0, 2.0, 3.0])
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        tau = a._effort_fallback(art, 3, np.zeros(3), measured_read=False, err=ValueError("shape"))
    assert art.calls == 1 and np.allclose(tau, [1.0, 2.0, 3.0])
    assert caplog.records[-1].getMessage().endswith("publishing measured efforts")


def test_effort_fallback_says_zeros_when_the_lazy_read_fails_too(caplog):
    a = _robot(kp=[35.0] * 3)
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        tau = a._effort_fallback(_EffortArt(fail=True), 3, np.ones(3), measured_read=False, err=ValueError("shape"))
    assert np.all(tau == 0.0)
    assert caplog.records[-1].getMessage().endswith("publishing zeros (measured read failed)")


def test_effort_fallback_keeps_an_already_measured_tau_and_warns_once(caplog):
    a = _robot(kp=[35.0] * 3)
    art = _EffortArt(tau=[9.0, 9.0, 9.0])
    measured = np.asarray([0.5, 0.6, 0.7])
    with caplog.at_level(logging.WARNING, logger="isaac_bridge"):
        tau = a._effort_fallback(art, 3, measured, measured_read=True, err=ValueError("shape"))
        a._effort_fallback(art, 3, measured, measured_read=True, err=ValueError("shape"))
    assert art.calls == 0 and tau is measured                    # no second round trip: the step already read it
    assert len([r for r in caplog.records if "drive-effort estimate unavailable" in r.getMessage()]) == 1
