"""BAM servo models: the params of every joint are validated before a model is attached.

A bad joint (invalid JSON, a non-numeric or out-of-range value, an unknown actuator family without firmware
constants, model "bam" without params) is rejected with one WARNING and keeps the PD drive; valid joints keep their
model; a setup failure leaves nothing attached, so no physics step ever runs a half-built model."""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest

from fakes import command, make_moderator, make_robot
from scripts.device import actuator_model

# BAM @62bd8ce bam/params/xl330/m6.json (the XL330 M6 model) and a servo-defaults block as a robot would author it
XL330_M6 = {
    "kt": 0.36601349688984386, "R": 2.8113923539223227, "armature": 0.0018077432831600838,
    "q_offset": 0.0271132870444849, "friction_base": 0.004771183165566, "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698, "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05, "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415, "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307, "alpha": 8.683259907618984, "friction_viscous": 0.005359668274599504,
    "model": "m6", "actuator": "xl330",
}
SERVO_DEFAULTS = {"kp_fw": 200.0, "vin": 7.4, "max_current": None, "vin_drop_gain": 0.1, "vin_min": 6.0,
                      "delay_steps": 0, "friction_scale": 1.0}


class _Attr:
    def __init__(self, v):
        self.v = v

    def IsValid(self):
        return self.v is not None

    def Get(self):
        return self.v


class _Prim:
    def __init__(self, path, attrs=None):
        self.path = path
        self.attrs = dict(attrs or {})

    def GetAttribute(self, name):
        return _Attr(self.attrs.get(name))

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self):
        return self.path

    def IsValid(self):
        return True

    def __bool__(self):
        return True


def _bam(params=XL330_M6, name="xl330/m6"):
    attrs = {"rc:actuator:model": "bam", "rc:actuator:name": name}
    if params is not None:
        attrs["rc:actuator:params"] = params if isinstance(params, str) else json.dumps(params)
    return attrs


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def warnings(self, needle=""):
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


@pytest.fixture
def logs():
    handler = _Records()
    for name in ("isaac_bridge", actuator_model.__name__):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    yield handler
    for name in ("isaac_bridge", actuator_model.__name__):
        logging.getLogger(name).removeHandler(handler)


def _robot(joint_attrs, root_attrs=None, dt=1 / 200, all_joints=None):
    """A robot on the raw-view fakes whose USD metadata (root + joint prims) is read on the next synchronize()."""
    n = len(joint_attrs)
    a, pv = make_robot(n, physics_dt=dt)
    root = _Prim("/W/r", root_attrs)
    joints = [_Prim(f"/W/r/joints/j{i}", attrs) for i, attrs in enumerate(joint_attrs)]
    a._root_prim = lambda: root
    a._joint_prims = lambda: joints
    a._all_joint_prims = lambda: (joints if all_joints is None else all_joints)
    a._meta_loaded = False
    return a, pv


def _step(a, pv, n, q_d=0.2):
    """One command + two physics steps; returns the last written (kp, kd, eff) arrays (None where not written)."""
    a.update_target("sim_command", command(n, [q_d] * n))
    out = {}
    for _ in range(2):
        pv.calls.clear()
        a.synchronize()
        for c in pv.calls:
            if isinstance(c, tuple):
                out[c[0]] = np.asarray(c[1]).reshape(-1)
    return out.get("kp"), out.get("kd"), out.get("eff")


def _mask(a):
    return None if a._am is None else a._am.mask.astype(int).tolist()


# ------------------------------------------------------------------ unchanged paths
def test_no_rc_actuator_keeps_the_pd_drive(logs):
    a, pv = _robot([{}, {}, {}])
    kp, kd, eff = _step(a, pv, 3)
    assert a._am is None
    assert kp is not None and np.all(kp > 0.0)          # the PhysX PD drive gains
    assert eff is None                                   # no servo efforts
    assert logs.warnings() == []


def test_partial_bam_models_only_the_bam_joints(logs):
    a, pv = _robot([_bam(), _bam(), {}], root_attrs={"rc:actuator:defaults": json.dumps(SERVO_DEFAULTS)})
    kp, kd, eff = _step(a, pv, 3)
    assert _mask(a) == [1, 1, 0]
    servo = a._am.groups[0][0]
    assert servo.kt == pytest.approx(XL330_M6["kt"]) and servo.kp_fw == 200.0 and servo.vin == 7.4
    assert kp[0] == kp[1] == 0.0 and kp[2] > 0.0         # servo DOFs: kp 0, torque from the model
    assert kd[0] == pytest.approx(servo.linear_damping, rel=1e-5)
    assert eff is not None and eff[0] != 0.0 and eff[2] == 0.0
    assert logs.warnings() == []


# ------------------------------------------------------------------ per-joint rejection
def test_invalid_json_on_one_joint_rejects_only_that_joint(logs):
    a, pv = _robot([_bam("{not json"), _bam(), {}])
    kp, kd, eff = _step(a, pv, 3)
    assert _mask(a) == [0, 1, 0]
    assert kp[0] > 0.0 and kp[1] == 0.0
    w = logs.warnings("rejected")
    assert len(w) == 1 and "/W/r/joints/j0" in w[0] and "j0" in w[0] and "not valid JSON" in w[0]
    assert "/W/r" in w[0] and "PD drive" in w[0]


def test_non_numeric_value_rejects_only_that_joint_not_the_robot(logs):
    a, pv = _robot([_bam(dict(XL330_M6, kt="abc")), _bam(), _bam()])
    kp, kd, eff = _step(a, pv, 3)
    assert _mask(a) == [0, 1, 1]
    assert kp[0] > 0.0 and kp[1] == kp[2] == 0.0
    w = logs.warnings("rejected")
    assert len(w) == 1 and "j0" in w[0] and "kt='abc' is not a number" in w[0]


def test_bam_without_params_is_rejected_instead_of_a_fake_servo(logs):
    a, pv = _robot([_bam(None), {}, {}])
    kp, kd, eff = _step(a, pv, 3)
    assert a._am is None and eff is None and np.all(kp > 0.0)
    w = logs.warnings("rejected")
    assert len(w) == 1 and "rc:actuator:params is missing" in w[0] and "j0" in w[0]


def test_unknown_family_without_firmware_constants_is_rejected(logs):
    a, pv = _robot([_bam(dict(XL330_M6, actuator="erob")), _bam(), {}])
    _step(a, pv, 3)
    assert _mask(a) == [0, 1, 0]
    w = logs.warnings("rejected")
    assert len(w) == 1 and "unknown actuator family 'erob'" in w[0] and "vin, kp_fw, error_gain missing" in w[0]


def test_unknown_family_with_explicit_firmware_constants_is_accepted_without_xl330_defaults(logs):
    params = dict(XL330_M6, actuator="erob", vin=24.0, kp_fw=10.0, error_gain=0.01)
    a, pv = _robot([_bam(params), {}, {}])
    _step(a, pv, 3)
    assert _mask(a) == [1, 0, 0]
    servo = a._am.groups[0][0]
    assert (servo.vin, servo.kp_fw, servo.error_gain, servo.max_pwm) == (24.0, 10.0, 0.01, 1.0)
    assert servo.max_current is None                      # not the XL330 1.75 A limiter
    assert logs.warnings() == []


def test_r_zero_is_rejected_before_attach_and_no_step_raises(logs):
    a, pv = _robot([_bam(dict(XL330_M6, R=0.0)), {}, {}])
    kp, kd, eff = _step(a, pv, 3)                         # used to raise ZeroDivisionError at every step
    assert a._am is None and eff is None and np.all(kp > 0.0)
    w = logs.warnings("rejected")
    assert len(w) == 1 and "R must be > 0" in w[0]


def test_setup_failure_leaves_no_half_attached_model(logs, monkeypatch):
    def boom(self):
        raise ZeroDivisionError("float division by zero")
    monkeypatch.setattr(actuator_model.ActuatorModelSet, "describe", boom)
    a, pv = _robot([_bam(), _bam(), {}])
    kp, kd, eff = _step(a, pv, 3)
    assert a._am is None and a._am_tau is None and a._am_joints is None
    assert eff is None and np.all(kp > 0.0)
    w = logs.warnings("setup failed")
    assert len(w) == 1 and "ZeroDivisionError" in w[0] and "PD drive on every DOF" in w[0]


def test_invalid_defaults_json_warns_and_is_ignored(logs):
    a, pv = _robot([_bam(dict(XL330_M6, kp_fw=200.0)), {}, {}], root_attrs={"rc:actuator:defaults": "{bad"})
    _step(a, pv, 3)
    assert _mask(a) == [1, 0, 0]
    w = logs.warnings("rc:actuator:defaults")
    assert len(w) == 1 and "/W/r" in w[0] and "ignored" in w[0]


def test_invalid_value_in_defaults_rejects_the_joints_it_reaches(logs):
    a, pv = _robot([_bam(), _bam(dict(XL330_M6, vin=7.0)), {}],
                   root_attrs={"rc:actuator:defaults": json.dumps({"vin": "7.4"})})
    _step(a, pv, 3)
    assert _mask(a) == [0, 1, 0]                          # j1 overrides vin with a number
    w = logs.warnings("rejected")
    assert len(w) == 1 and "vin='7.4' is not a number" in w[0]


def test_unknown_model_string_warns_and_keeps_pd(logs):
    a, pv = _robot([{"rc:actuator:model": "bma"}, {"rc:actuator:model": "pd"}, {}])
    _step(a, pv, 3)
    assert a._am is None
    w = logs.warnings()
    assert len(w) == 1 and "unknown rc:actuator:model 'bma'" in w[0]


def test_joint_count_mismatch_warns_only_for_a_bam_robot(logs):
    a, pv = _robot([_bam(), {}, {}])
    a._joint_prims = lambda: [_Prim("/W/r/joints/j0", _bam())]              # 1 actuated joint vs 3 DOF
    _step(a, pv, 3)
    assert a._am is None
    w = logs.warnings("skipped")
    assert len(w) == 1 and "1 actuated USD joints vs 3 articulation DOF" in w[0]

    b, pv = _robot([{}, {}, {}])
    b._joint_prims = lambda: [_Prim("/W/r/joints/j0")]
    logs.records.clear()
    _step(b, pv, 3)
    assert b._am is None and logs.warnings() == []                        # robots without BAM stay quiet


# ------------------------------------------------------------------ physics rate warning
def test_long_physics_step_warns_once_without_changing_behaviour(logs):
    a, pv = _robot([_bam(), {}, {}], dt=1 / 100)
    _step(a, pv, 3)
    _step(a, pv, 3)
    assert _mask(a) == [1, 0, 0]                                          # the model still drives
    w = logs.warnings("Nyquist")
    assert len(w) == 1 and "10.00 ms" in w[0]

    b, pv = _robot([_bam(), {}, {}], dt=1 / 200)
    logs.records.clear()
    _step(b, pv, 3)
    assert logs.warnings("Nyquist") == []


# ------------------------------------------------------------------ validate_params / ActuatorModelSet
@pytest.mark.parametrize("patch, reason", [
    ({"kt": True}, "kt=True is not a number"),
    ({"kt": float("nan")}, "kt=nan is not finite"),
    ({"friction_base": float("inf")}, "friction_base=inf is not finite"),
    ({"R": -1.0}, "R must be > 0"),
    ({"kt": 0}, "kt must be > 0"),
    ({"kp_fw": None}, "kp_fw is null"),
    ({"dtheta_stribeck": 0.0}, "dtheta_stribeck must be > 0"),
    ({"armature": -1e-3}, "armature must be >= 0"),
    ({"delay_steps": -1}, "delay_steps must be >= 0"),
    ({"vin": 0.0}, "vin must be > 0"),
    ({"max_pwm": 0.0}, "max_pwm must be > 0"),
    ({"actuator": None}, "unknown actuator family '(none)'"),
])
def test_validate_params_rejects(patch, reason):
    with pytest.raises(actuator_model.ActuatorParamsError, match=None) as err:
        actuator_model.validate_params(dict(XL330_M6, **patch))
    assert reason in str(err.value)


def test_validate_params_rejects_a_missing_kt_and_a_non_object():
    params = dict(XL330_M6)
    del params["kt"]
    with pytest.raises(actuator_model.ActuatorParamsError, match="kt is missing"):
        actuator_model.validate_params(params)
    with pytest.raises(actuator_model.ActuatorParamsError, match="JSON object"):
        actuator_model.validate_params([1, 2])


def test_validate_params_accepts_presets_and_nullable_keys():
    actuator_model.validate_params(XL330_M6)
    actuator_model.validate_params(dict(XL330_M6, **SERVO_DEFAULTS))                   # max_current null
    actuator_model.validate_params(dict(XL330_M6, max_velocity=None, vin_min=None, lift_velocity_clamp=True,
                                        inertia_source="mass_matrix", q_offset=0.03, delay_steps=4))
    for family in actuator_model.FAMILIES:
        actuator_model.validate_params({"kt": 1.0, "R": 2.0, "actuator": family.upper(), "model": "m1"})


def test_model_set_leaves_out_an_invalid_spec_instead_of_raising():
    am = actuator_model.ActuatorModelSet([dict(XL330_M6, R=0.0), dict(XL330_M6), None], 1 / 200)
    assert am.mask.tolist() == [False, True, False]
    assert am.rejected == [([0], "R must be > 0 (got 0.0)")]
    assert np.all(np.isfinite(am.dampings()))
    with pytest.raises(actuator_model.ActuatorParamsError):
        actuator_model.BamServo(dict(XL330_M6, kt="abc"), 1, 1 / 200)


def test_read_actuator_specs_without_a_logger_warns_on_the_module_logger(logs):
    root = _Prim("/W/r")
    specs = actuator_model.read_actuator_specs(root, [_Prim("/W/r/j0", _bam("[1]")), _Prim("/W/r/j1", _bam())])
    assert specs[0] is None and specs[1]["_name"] == "xl330/m6"
    w = logs.warnings("rejected")
    assert len(w) == 1 and "must be a JSON object, got list" in w[0]


# ------------------------------------------------------------------ through the moderator
def test_bam_robot_with_r_zero_steps_through_the_moderator_without_errors(logs):
    a, pv = _robot([_bam(dict(XL330_M6, R=0.0)), _bam(), {}])
    a.update_target("sim_command", command(3, [0.2] * 3))
    m = make_moderator({"duck": a})
    for _ in range(3):
        m.synchronize(1 / 200)
    assert logs.warnings("sync error") == []
    assert _mask(a) == [0, 1, 0]
    assert math.isfinite(float(a._am.dampings()[1]))
