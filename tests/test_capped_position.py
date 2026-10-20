"""The capped-P position drive (rc:positionDriveModel = "capped_p_passive") keeps its torque bounds and leaves the
other command modes alone."""
import numpy as np

from fakes import command, make_robot
from scripts import protocol


def _device():
    d, _ = make_robot(3)
    d._mode = np.array([protocol.MODE_POSITION, protocol.MODE_POSITION, protocol.MODE_TORQUE], dtype=np.uint8)
    d._capped_position = dict(limits=np.array([56., 56., 12.]), friction=np.array([.5, .5, .1]))
    return d


def _capped_robot():
    d, pv = make_robot(3)
    d._usd_kp = np.full(3, 100., dtype=np.float32)
    d._usd_kd = np.full(3, 18., dtype=np.float32)
    d._usd_vel_ff = False
    d._capped_position = dict(limits=np.full(3, 56.), friction=np.full(3, .5))
    return d, pv


def test_drive_saturates_p_before_losses_and_caps_total_braking():
    d = _device()
    q = np.zeros(3)
    dq = np.array([.75, -9., 2.])
    kp = np.array([100., 100., 3.])
    kd = np.array([18., 18., 4.])
    vel = np.array([0., 0., 5.])
    pos = np.array([2., 2., 6.])
    eff = np.array([0., 0., 7.])
    d._apply_capped_position(q, dq, kp, kd, vel, pos, eff)
    # the powered torque saturates at 56 Nm; then 13.5 Nm of damping and 0.5 Nm of friction are deducted
    np.testing.assert_allclose(eff[:2], [42., 56.], atol=1e-8)
    np.testing.assert_array_equal(kp, [0., 0., 3.])
    np.testing.assert_array_equal(kd, [0., 0., 4.])
    np.testing.assert_array_equal(vel, [0., 0., 5.])
    assert eff[2] == 7. and pos[2] == 6.


def test_drive_recomputes_from_feedback_and_follows_the_latched_modes():
    d = _device()
    d._latch_enabled = True
    modes = np.array([protocol.MODE_POSITION, protocol.MODE_TORQUE, protocol.MODE_TORQUE], dtype=np.uint8)
    d._latched = (modes,) + tuple(np.zeros(3, np.float32) for _ in range(5))

    def evaluate(velocity):
        kp = np.full(3, 100.)
        kd = np.full(3, 18.)
        eff = np.zeros(3)
        d._apply_capped_position(np.zeros(3), np.full(3, velocity), kp, kd, np.zeros(3), np.ones(3), eff)
        assert kp[1] == 100. and eff[1] == 0.
        return eff[0]

    assert evaluate(1.) < evaluate(0.)


def test_feedback_refresh_and_idle_clear_reach_physx_without_new_targets():
    d, pv = _capped_robot()
    d.update_target("sim_command", command(3, [2., 2., 2.]))
    d.synchronize()
    np.testing.assert_allclose(pv.last("eff"), 56.)
    assert not d._cmd_dirty
    pv.dq[:] = .75
    d.synchronize()
    np.testing.assert_allclose(pv.last("eff"), 42.)
    # losing every owned command must clear PhysX's persistent actuation force
    d._mode[:] = protocol.MODE_IDLE
    d._latched = None
    d.synchronize()
    np.testing.assert_array_equal(pv.last("eff"), np.zeros((1, 3)))
    assert not d._drive_has_effort


def test_published_effort_is_the_applied_motor_torque_not_the_joint_reaction():
    # A kp-zeroed capped DOF would otherwise publish the measured joint reaction torque, whose foot-impact spikes
    # far exceed the actuator limit.
    d, pv = _capped_robot()
    d._art.get_measured_joint_efforts = lambda: np.full((1, 3), 300., dtype=np.float32)
    d.update_target("sim_command", command(3, [2., 2., 2.]))
    d.synchronize()
    np.testing.assert_allclose(d._actual[2], 56.)
    pv.dq[:] = .75
    d.synchronize()
    np.testing.assert_allclose(d._actual[2], 42.)
    # without an owned command the capped model does not drive: nothing stale is published as its torque
    d._mode[:] = protocol.MODE_IDLE
    d._latched = None
    d.synchronize()
    assert d._capped_tau is None
