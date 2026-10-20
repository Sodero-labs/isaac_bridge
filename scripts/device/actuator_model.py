"""Servo actuator models: BAM (Better Actuator Model, Rhoban) voltage / friction servos.

Why: a PhysX joint drive is an ideal spring-damper. Policies for small servo robots may be trained against a BAM
model instead (the Microduck policies, pollen-robotics/microduck_rl, against the M6 model of the Dynamixel XL330):
firmware P loop -> PWM duty -> DC motor with back-EMF (the torque falls with speed and saturates with the battery
voltage), gearbox friction that grows with the motor torque (about 27 % of it on the XL330), Stribeck stiction,
viscous drag, battery sag under load and a command delay. Running the same model inside Isaac Sim removes that part
of the sim-to-sim and sim-to-real gap.

USD contract (read by device/articulation.py):

    joint prim   rc:actuator:model    string  "bam"  (missing or "pd": the PhysX PD drive)
                 rc:actuator:name     string  display name, e.g. "xl330/m6"
                 rc:actuator:params   string  JSON: a BAM parameter file (kt, R, armature, friction_*, model,
                                              actuator) plus optional servo settings (SERVO_KEYS below)
    root prim    rc:actuator:defaults string  JSON of servo settings shared by every modelled joint of the
                                              articulation (the joint's own params win)

Every "bam" joint is validated before a model is built (validate_params): its params must be a JSON object whose
numeric keys are finite JSON numbers, with kt > 0, R > 0 and either a known actuator family (FAMILIES) or explicit
firmware constants (FIRMWARE_KEYS). A joint that fails keeps the PhysX PD drive and is named in one WARNING; the
other joints keep their models.

The torque pipeline mirrors bam.mujoco.MujocoController / bam.mjlab.BamActuator at commit 62bd8ce (the one
microduck_rl pins). One deliberate difference: MuJoCo applies the friction budget through its constraint solver
(dof_frictionloss / dof_damping rewritten every step, BAM "Algorithm 1"); PhysX has no per-step friction budget,
so BamServo applies the budget as an explicit stopping torque (see BamServo.compute). Pure numpy: no Isaac, no
MuJoCo, no bam package needed at runtime, so it is testable outside the simulator.
"""
from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Sequence, Tuple

ATTR_MODEL = "rc:actuator:model"
ATTR_NAME = "rc:actuator:name"
ATTR_PARAMS = "rc:actuator:params"
ATTR_DEFAULTS = "rc:actuator:defaults"
MODEL_BAM = "bam"

# Firmware/electrical constants of the BAM VoltageControlledActuator families
# (bam/dynamixel/actuator.py, bam/feetech/actuator.py). error_gain turns kp_fw x position error [rad] into a PWM
# duty cycle; it was measured with an oscilloscope by the BAM authors. A params JSON may override any of them.
_XL330_ERROR_GAIN = (4096.0 / (2.0 * math.pi)) / (256.0 * 885.0)   # encoder counts per rad / (kp divisor x PWM limit)
FAMILIES = {
    "xl330":   dict(vin=7.5,  kp_fw=400.0, error_gain=_XL330_ERROR_GAIN, max_pwm=1.0,    max_current=1.75),
    "xl320":   dict(vin=7.5,  kp_fw=32.0,  error_gain=0.05048199,       max_pwm=1.0,    max_current=None),
    "mx64":    dict(vin=15.0, kp_fw=32.0,  error_gain=0.158,            max_pwm=0.9625, max_current=None),
    "mx106":   dict(vin=15.0, kp_fw=32.0,  error_gain=0.158,            max_pwm=0.9625, max_current=None),
    "sts3215": dict(vin=7.4,  kp_fw=32.0,  error_gain=0.166,            max_pwm=0.97,   max_current=None,
                    max_velocity=(3400.0 * 2.0 * math.pi) / 4096.0),   # internal target slew (firmware)
}
# Servo settings a params JSON / rc:actuator:defaults may carry (on top of the identified BAM parameters).
SERVO_KEYS = ("kp_fw", "vin", "error_gain", "max_pwm", "max_current", "max_velocity", "vin_drop_gain", "vin_min",
              "delay_steps", "friction_scale", "kd_scale", "stiction_inertia_scale", "lift_velocity_clamp",
              "stiction_velocity", "inertia_source")
# stribeck, load_dependent, directional, quadratic  (bam/model.py `models`)
MODEL_FLAGS = {
    "m1": (False, False, False, False),
    "m2": (True, False, False, False),
    "m3": (False, True, False, False),
    "m4": (True, True, False, False),
    "m5": (True, True, True, False),
    "m6": (True, True, True, True),
}

# -- params validation ----------------------------------------------------------------------------------------
# Numeric keys of a params JSON (identified BAM parameters + servo settings). When present each must be a finite
# JSON number (not a string, not a bool); NULLABLE_KEYS may also be null (= that feature off).
BAM_NUMERIC_KEYS = ("kt", "R", "armature", "friction_base", "friction_viscous", "friction_stribeck",
                    "dtheta_stribeck", "alpha", "load_friction_base", "load_friction_stribeck", "load_friction_motor",
                    "load_friction_external", "load_friction_motor_stribeck", "load_friction_external_stribeck",
                    "load_friction_motor_quad", "load_friction_external_quad")
SERVO_NUMERIC_KEYS = ("kp_fw", "vin", "error_gain", "max_pwm", "max_current", "max_velocity", "vin_drop_gain",
                      "vin_min", "delay_steps", "friction_scale", "kd_scale", "stiction_inertia_scale",
                      "stiction_velocity")
NULLABLE_KEYS = ("max_current", "max_velocity", "vin_min")
# What a params JSON must spell out when its "actuator" family is not one of FAMILIES (max_pwm defaults to 1.0,
# max_current / max_velocity to off).
FIRMWARE_KEYS = ("vin", "kp_fw", "error_gain")

_LOG = logging.getLogger(__name__)


class ActuatorParamsError(ValueError):
    """A joint's servo params cannot drive a model; the message is the reason (the joint keeps the PD drive)."""


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_params(params) -> None:
    """Raise ActuatorParamsError(reason) unless `params` (joint params merged over rc:actuator:defaults) can
    build a BamServo: numeric keys finite JSON numbers, kt > 0, R > 0, a known family or explicit firmware
    constants, and no value that would divide by zero or reverse a sign in the torque pipeline."""
    if not isinstance(params, dict):
        raise ActuatorParamsError(f"params must be a JSON object, got {type(params).__name__}")
    for key in BAM_NUMERIC_KEYS + SERVO_NUMERIC_KEYS:
        if key not in params:
            continue
        v = params[key]
        if v is None:
            if key in NULLABLE_KEYS:
                continue
            raise ActuatorParamsError(f"{key} is null")
        if not _number(v):
            raise ActuatorParamsError(f"{key}={v!r} is not a number")
        if not math.isfinite(float(v)):
            raise ActuatorParamsError(f"{key}={v!r} is not finite")
    for key in ("kt", "R"):
        if params.get(key) is None:
            raise ActuatorParamsError(f"{key} is missing")
        if float(params[key]) <= 0.0:
            raise ActuatorParamsError(f"{key} must be > 0 (got {params[key]!r})")
    for key in ("armature", "delay_steps", "stiction_velocity"):
        if params.get(key) is not None and float(params[key]) < 0.0:
            raise ActuatorParamsError(f"{key} must be >= 0 (got {params[key]!r})")
    if params.get("dtheta_stribeck") is not None and float(params["dtheta_stribeck"]) <= 0.0:
        raise ActuatorParamsError(f"dtheta_stribeck must be > 0 (got {params['dtheta_stribeck']!r})")
    family = str(params.get("actuator") or "").strip().lower()
    if family in FAMILIES:
        firmware = dict(FAMILIES[family])
    else:
        missing = [k for k in FIRMWARE_KEYS if params.get(k) is None]
        if missing:
            raise ActuatorParamsError(
                f"unknown actuator family {family or '(none)'!r} (known: {', '.join(sorted(FAMILIES))}) and no "
                f"explicit firmware constants ({', '.join(missing)} missing)")
        firmware = {}
    firmware.update({k: params[k] for k in SERVO_NUMERIC_KEYS if k in params})
    for key in ("vin", "error_gain"):
        if not float(firmware.get(key) or 0.0) > 0.0:
            raise ActuatorParamsError(f"{key} must be > 0 (got {firmware.get(key)!r})")
    if firmware.get("max_pwm") is not None and float(firmware["max_pwm"]) <= 0.0:
        raise ActuatorParamsError(f"max_pwm must be > 0 (got {firmware['max_pwm']!r})")
    if firmware.get("kp_fw") is not None and float(firmware["kp_fw"]) < 0.0:
        raise ActuatorParamsError(f"kp_fw must be >= 0 (got {firmware['kp_fw']!r})")


def _f(p: dict, key: str, default: float = 0.0) -> float:
    v = p.get(key)
    return float(default if v is None else v)


class BamServo:
    """One BAM servo model shared by `n` joints (same parameters), stepped at the physics rate.

    compute() returns the torque to apply on each joint = motor torque (firmware P loop, PWM/voltage saturation,
    back-EMF) + friction (velocity-independent budget applied as a stopping torque, + viscous). State: the
    previous step's motor torque (the load the gearbox friction and the battery sag are computed from, one
    step late exactly like the reference), the command delay line and the STS3215 internal target.
    """

    def __init__(self, params: dict, n: int, dt: float):
        validate_params(params)        # ActuatorParamsError: the caller keeps the PD drive for these joints
        p = dict(params)
        self.family = str(p.get("actuator", "") or "").strip().lower()
        # an unknown family carries its own firmware constants (validate_params made sure of it)
        base = dict(FAMILIES.get(self.family, {}))
        for k in SERVO_KEYS:
            if k in p:
                base[k] = p[k]
        self.n = int(n)
        self.dt = float(dt)
        self.model = str(p.get("model", "m1") or "m1").lower()
        self.name = str(p.get("_name") or p.get("name") or f"{self.family or '?'}/{self.model}")
        flags = MODEL_FLAGS.get(self.model)
        if flags is None:   # unknown variant: infer from the keys present
            flags = ("friction_stribeck" in p, any(k.startswith("load_friction") for k in p),
                     "load_friction_motor" in p, "load_friction_motor_quad" in p)
        self.stribeck, self.load_dependent, self.directional, self.quadratic = flags

        # electrical / mechanical
        self.kt = _f(p, "kt", 1.0)
        self.R = _f(p, "R", 1.0)
        self.armature = _f(p, "armature", 0.0)
        # friction
        self.friction_base = _f(p, "friction_base")
        self.friction_viscous = _f(p, "friction_viscous")
        self.friction_stribeck = _f(p, "friction_stribeck")
        self.dtheta_stribeck = _f(p, "dtheta_stribeck", 1.0)
        self.alpha = _f(p, "alpha", 1.0)
        self.load_friction_base = _f(p, "load_friction_base")
        self.load_friction_stribeck = _f(p, "load_friction_stribeck")
        self.load_friction_motor = _f(p, "load_friction_motor")
        self.load_friction_external = _f(p, "load_friction_external")
        self.load_friction_motor_stribeck = _f(p, "load_friction_motor_stribeck")
        self.load_friction_external_stribeck = _f(p, "load_friction_external_stribeck")
        self.load_friction_motor_quad = _f(p, "load_friction_motor_quad")
        self.load_friction_external_quad = _f(p, "load_friction_external_quad")
        # firmware / supply
        self.kp_fw = float(base.get("kp_fw") or 0.0)
        self.error_gain = float(base.get("error_gain") or 0.0)
        self.max_pwm = float(base.get("max_pwm") or 1.0)
        mc = base.get("max_current")
        self.max_current = None if (mc is None or float(mc) <= 0.0) else float(mc)
        mv = base.get("max_velocity")
        self.max_velocity = None if (mv is None or float(mv) <= 0.0) else float(mv)
        self.vin = float(base.get("vin") or 0.0)
        self.vin_drop_gain = float(base.get("vin_drop_gain") or 0.0)        # [V/Nm] x sum |motor torque|
        vm = base.get("vin_min")
        self.vin_min = None if vm is None else float(vm)
        self.delay_steps = max(0, int(base.get("delay_steps") or 0))       # physics steps
        self.friction_scale = float(base.get("friction_scale") if base.get("friction_scale") is not None else 1.0)
        self.kd_scale = float(base.get("kd_scale") if base.get("kd_scale") is not None else 1.0)
        self.stiction_inertia_scale = float(base.get("stiction_inertia_scale")
                                            if base.get("stiction_inertia_scale") is not None else 1.0)
        # PhysX only: keep the USD joint velocity clamp (default) or lift it. Lifted, a light link can spin up to
        # hundreds of rad/s on a contact impulse, and the back-EMF term then brakes with torques far above the rating.
        self.lift_velocity_clamp = bool(base.get("lift_velocity_clamp") or False)
        # Dissipative friction band [rad/s] used when the joint's apparent inertia is unknown (see compute()).
        sv = base.get("stiction_velocity")
        self.stiction_velocity = float(sv) if sv else 0.05
        # "none" (default) = dissipative band friction; "mass_matrix" = let the bridge feed the joint's apparent inertia
        # (PhysX mass-matrix diagonal) for the MuJoCo-style stopping torque (unverified in Isaac, opt-in).
        self.inertia_source = str(base.get("inertia_source") or "none").lower()
        # The velocity-proportional part (back-EMF kt^2/R + viscous friction) can be applied IMPLICITLY by the
        # simulator (PhysX drive damping, MuJoCo dof_damping) instead of as an explicit torque: numerically far more
        # robust against contact impulses on light links. compute() then leaves it out; see implicit_damping.
        self.implicit_damping = False
        self.reset()

    # -- derived (for logs / sanity) ----------------------------------------
    @property
    def stall_torque(self) -> float:
        return self.kt * self.vin * self.max_pwm / self.R

    @property
    def no_load_speed(self) -> float:
        return self.vin * self.max_pwm / self.kt if self.kt > 0 else float("inf")

    @property
    def linear_stiffness(self) -> float:
        """Small-error equivalent of the firmware loop [Nm/rad] (= the MJCF fallback kp)."""
        return self.kt / self.R * self.vin * self.kp_fw * self.error_gain

    @property
    def linear_damping(self) -> float:
        """Back-EMF + viscous friction [Nm s/rad] (= the MJCF fallback joint damping)."""
        return self.kt * self.kt / self.R * self.kd_scale + self.friction_viscous

    def describe(self) -> str:
        cur = f"{self.max_current:.2f} A" if self.max_current is not None else "off"
        return (f"bam {self.model} {self.family or '?'} x{self.n}: kt {self.kt:.3f} R {self.R:.2f} armature {self.armature:.2e} "
                f"vin {self.vin:.2f} V kp_fw {self.kp_fw:g} (k_lin {self.linear_stiffness:.3f} Nm/rad, d_lin "
                f"{self.linear_damping:.4f} Nm s/rad, stall {self.stall_torque:.2f} Nm, no-load {self.no_load_speed:.1f} rad/s) "
                f"friction base {self.friction_base:.4f} stribeck {self.friction_stribeck:.4f} visc {self.friction_viscous:.4f} "
                f"load motor {self.load_friction_motor + self.load_friction_base:.3f} ext {self.load_friction_external:.3f} "
                f"x{self.friction_scale:g}; current limit {cur}; vin drop {self.vin_drop_gain:g} V/Nm (min {self.vin_min}); "
                f"delay {self.delay_steps} step(s)")

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.prev_tau_m = np.zeros(self.n, dtype=np.float64)
        self._delay = None          # (delay_steps + 1, n) ring of q_target, oldest first
        self._q_smooth = None       # STS3215 internal (slew-limited) target
        self.last_vin = self.vin
        self.last_budget = np.zeros(self.n, dtype=np.float64)
        self.last_tau_drive = np.zeros(self.n, dtype=np.float64)

    # -- BAM friction budget (bam/model.py Model.compute_frictions, velocity-independent part) --------------
    def friction_budget(self, motor_torque: np.ndarray, external_torque: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
        if self.stribeck:
            s = np.exp(-np.abs(dtheta / self.dtheta_stribeck) ** self.alpha)
        else:
            s = np.zeros_like(dtheta)
        fl = np.full_like(motor_torque, self.friction_base)
        if self.directional:
            gearbox = np.abs(external_torque * self.load_friction_external - motor_torque * self.load_friction_motor)
            if self.stribeck:
                gearbox_s = np.abs(external_torque * self.load_friction_external_stribeck
                                   - motor_torque * self.load_friction_motor_stribeck)
        else:
            gearbox = np.abs(external_torque - motor_torque)
        if self.load_dependent:
            fl = fl + (gearbox if self.directional else self.load_friction_base * gearbox)
        if self.stribeck:
            fl = fl + s * self.friction_stribeck
            if self.load_dependent:
                if self.directional:
                    fl = fl + gearbox_s * s
                else:
                    fl = fl + self.load_friction_stribeck * gearbox * s
                if self.quadratic:
                    enable_q = (np.sign(external_torque) != np.sign(motor_torque)).astype(np.float64)
                    abs_ext, abs_mot = np.abs(external_torque), np.abs(motor_torque)
                    drive = (abs_ext < abs_mot).astype(np.float64)
                    backdrive = (abs_ext > abs_mot).astype(np.float64)
                    fl = fl + s * (drive * self.load_friction_external_quad * abs_ext ** 2
                                   + backdrive * self.load_friction_motor_quad * abs_mot ** 2) * enable_q
        return fl

    # -- per physics step ---------------------------------------------------
    def compute(self, q_target, q, dq, enable=None, ext=None, inertia=None) -> "Tuple[np.ndarray, np.ndarray]":
        """Torque to apply on the joints for this physics step.

        q_target, q, dq: (n,) [rad], [rad], [rad/s].  enable: (n,) bool, False = torque disabled (unpowered servo:
        no motor torque, no back-EMF braking, friction still acts).  ext: (n,) estimate of the external torque on
        the joint [Nm] (gravity/Coriolis/contacts, BAM sign convention: -qfrc_bias + qfrc_constraint); None = 0.
        inertia: (n,) apparent inertia of each joint [kg m^2] (mass-matrix diagonal) — enables the MuJoCo-style
        stopping-torque friction; None = the dissipative band form (no inertia knowledge needed).
        Returns (tau_applied, tau_motor).
        """
        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        n = self.n
        # command delay (training: 3-6 physics steps between the policy output and the motor)
        if self.delay_steps > 0:
            if self._delay is None:
                self._delay = np.repeat(q_target[None, :], self.delay_steps + 1, axis=0)
            self._delay = np.roll(self._delay, -1, axis=0)
            self._delay[-1] = q_target
            q_target = self._delay[0]
        # STS3215: the firmware slews its internal target at max_velocity
        if self.max_velocity is not None:
            if self._q_smooth is None:
                self._q_smooth = q.copy()
            step = self.max_velocity * self.dt
            self._q_smooth = np.clip(q_target, self._q_smooth - step, self._q_smooth + step)
            q_target = self._q_smooth
        # supply voltage with the load-dependent sag (current draw ~ previous motor torques)
        vin = self.vin
        if self.vin_drop_gain > 0.0:
            vin = vin - self.vin_drop_gain * float(np.sum(np.abs(self.prev_tau_m)))
            if self.vin_min is not None:
                vin = max(vin, self.vin_min)
        self.last_vin = vin
        # firmware P loop -> duty cycle (optional current limiter as a duty window), PWM saturation -> volts
        dq_e = dq * self.kd_scale
        duty = (q_target - q) * self.kp_fw * self.error_gain
        if self.max_current is not None and vin > 0.0:
            center = self.kt * dq_e / vin
            span = self.R * self.max_current / vin
            duty = np.clip(duty, center - span, center + span)
        duty = np.clip(duty, -self.max_pwm, self.max_pwm)
        volts = vin * duty
        tau_emf = -self.kt * self.kt * dq_e / self.R                       # back-EMF braking
        tau_m = self.kt * volts / self.R + (0.0 if self.implicit_damping else tau_emf)
        if enable is not None:
            tau_m = np.where(np.asarray(enable, dtype=bool).reshape(-1), tau_m, 0.0)
        # friction budget from the PREVIOUS motor torque (lagged one step, as in the reference)
        ext_a = np.zeros(n, dtype=np.float64) if ext is None else np.asarray(ext, dtype=np.float64).reshape(-1)
        budget = self.friction_budget(self.prev_tau_m, ext_a, dq) * self.friction_scale
        self.last_budget = budget
        drive = tau_m + ext_a
        if inertia is not None:
            # MuJoCo frictionloss semantics (BAM Algorithm 1): the torque that brings the joint to rest this step
            # against the other torques, clipped to the budget — valid only with the joint's real apparent inertia.
            i_eff = np.asarray(inertia, dtype=np.float64).reshape(-1) * self.stiction_inertia_scale
            tau_f = np.clip(-(dq * i_eff / self.dt + drive), -budget, budget)
        else:
            # Dissipative band form: above stiction_velocity the friction opposes the motion, below it the friction
            # cancels the driving torque (stiction). It never pushes along the motion. (Predicting a within-step
            # reversal from the rotor inertia alone would make the friction ASSIST a braked joint whenever
            # |tau_m| dt / armature > |dq|.)
            w = np.clip(np.abs(dq) / self.stiction_velocity, 0.0, 1.0)
            tau_f = -(1.0 - w) * np.clip(drive, -budget, budget) - w * budget * np.sign(dq)
        tau_v = np.zeros(n, dtype=np.float64) if self.implicit_damping else -self.friction_viscous * dq
        # The friction budget (next step) sees the full electrical torque incl. back-EMF, like MuJoCo's qfrc_actuator.
        tau_motor = tau_m if not self.implicit_damping else tau_m + (np.where(np.asarray(enable, dtype=bool).reshape(-1), tau_emf, 0.0) if enable is not None else tau_emf)
        self.prev_tau_m = tau_motor.copy()
        # Reported torque = what the H-bridge drives, kt V / R (bounded by the stall torque). The back-EMF braking
        # of a fast back-driven joint is real but not bounded by the servo's rating (reported, it would trip a
        # controller's torque supervision); the simulator applies it as damping.
        self.last_tau_drive = self.kt * volts / self.R if enable is None else np.where(np.asarray(enable, dtype=bool).reshape(-1), self.kt * volts / self.R, 0.0)
        return tau_m + tau_f + tau_v, self.last_tau_drive


class ActuatorModelSet:
    """Per-DOF actuator models of one articulation (canonical/actuator order). DOFs without a model keep the
    PhysX drive; the others are grouped by identical parameters into vectorized BamServo instances. A spec that
    fails validate_params is left out (its DOFs keep the PhysX drive) and listed in `rejected` as
    (DOF indices, reason) — read_actuator_specs has normally rejected and logged it already."""

    def __init__(self, specs: "Sequence[Optional[dict]]", dt: float):
        self.n = len(specs)
        self.dt = float(dt)
        self.mask = np.zeros(self.n, dtype=bool)
        self.groups: "List[Tuple[BamServo, np.ndarray]]" = []
        self.rejected: "List[Tuple[List[int], str]]" = []
        by_key: "Dict[str, List[int]]" = {}
        keyed: "Dict[str, dict]" = {}
        for i, spec in enumerate(specs):
            if not spec:
                continue
            key = json.dumps(spec, sort_keys=True, default=str)
            by_key.setdefault(key, []).append(i)
            keyed[key] = spec
        for key, idx in by_key.items():
            arr = np.asarray(idx, dtype=np.int64)
            try:
                servo = BamServo(keyed[key], len(arr), self.dt)
            except ActuatorParamsError as e:
                self.rejected.append((list(idx), str(e)))
                continue
            self.groups.append((servo, arr))
            self.mask[arr] = True
        self.tau_motor = np.zeros(self.n, dtype=np.float64)

    @property
    def active(self) -> bool:
        return bool(self.groups)

    def set_implicit_damping(self, on: bool) -> None:
        for servo, _ in self.groups:
            servo.implicit_damping = bool(on)

    def dampings(self) -> np.ndarray:
        """(n,) implicit damping per DOF [Nm s/rad] = back-EMF kt^2/R x kd_scale + viscous friction; 0 where no model."""
        out = np.zeros(self.n, dtype=np.float64)
        for servo, idx in self.groups:
            out[idx] = servo.linear_damping
        return out

    def wants_inertia(self) -> bool:
        return any(s.inertia_source == "mass_matrix" for s, _ in self.groups)

    def lift_velocity_clamp(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=bool)
        for servo, idx in self.groups:
            out[idx] = servo.lift_velocity_clamp
        return out

    def reset(self) -> None:
        for servo, _ in self.groups:
            servo.reset()

    def armatures(self) -> np.ndarray:
        """(n,) model armature per DOF, NaN where no model."""
        out = np.full(self.n, np.nan, dtype=np.float64)
        for servo, idx in self.groups:
            out[idx] = servo.armature
        return out

    def compute(self, q_target, q, dq, enable=None, ext=None, inertia=None) -> np.ndarray:
        """(n,) torque for the modelled DOFs (0 elsewhere). Inputs in canonical order, (n,)."""
        out = np.zeros(self.n, dtype=np.float64)
        self.tau_motor[:] = 0.0
        for servo, idx in self.groups:
            tau, tau_m = servo.compute(
                np.asarray(q_target, dtype=np.float64)[idx], np.asarray(q, dtype=np.float64)[idx],
                np.asarray(dq, dtype=np.float64)[idx],
                None if enable is None else np.asarray(enable, dtype=bool)[idx],
                None if ext is None else np.asarray(ext, dtype=np.float64)[idx],
                None if inertia is None else np.asarray(inertia, dtype=np.float64)[idx])
            out[idx] = tau
            self.tau_motor[idx] = tau_m
        return out

    def describe(self) -> str:
        return "; ".join(f"{s.name} on DOF {idx.tolist()}: {s.describe()}" for s, idx in self.groups)


# -- USD side (pxr only inside) ---------------------------------------------
def _attr(prim, name):
    try:
        a = prim.GetAttribute(name)
        return a.Get() if a and a.IsValid() else None
    except Exception:
        return None


def _prim_path(prim) -> str:
    try:
        return str(prim.GetPath())
    except Exception:
        try:
            return str(prim.GetName())
        except Exception:
            return "?"


def _prim_name(prim) -> str:
    try:
        return str(prim.GetName())
    except Exception:
        return _prim_path(prim).rsplit("/", 1)[-1]


def read_actuator_specs(root_prim, joint_prims, log: "Optional[logging.Logger]" = None,
                        owner: str = "") -> "List[Optional[dict]]":
    """rc:actuator:* of each joint prim (canonical order) merged over the root's rc:actuator:defaults.
    Returns one params dict per joint (None = no model: PhysX PD drive). Never raises. Every "bam" joint is
    validated (validate_params); a joint whose params are missing, not JSON, not an object or invalid gets None
    and one WARNING on `log` naming the articulation, the joint prim and the reason. Invalid defaults JSON is
    warned about and ignored; an rc:actuator:model other than "bam" / "pd" is warned about (PD drive)."""
    log = log if log is not None else _LOG
    tag = f"[{owner}] " if owner else ""
    root_path = _prim_path(root_prim) if root_prim is not None else "?"
    defaults = {}
    raw = _attr(root_prim, ATTR_DEFAULTS)
    if raw:
        try:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
            defaults = parsed
        except Exception as e:
            log.warning(f"{tag}{ATTR_DEFAULTS} on {root_path} is invalid ({type(e).__name__}: {e}) -> ignored; "
                        f"each servo joint uses its own params only")
            defaults = {}
    specs = []
    for prim in joint_prims:
        model = _attr(prim, ATTR_MODEL)
        model_s = "" if model is None else str(model).strip().lower()
        if model_s != MODEL_BAM:
            if model_s not in ("", "pd"):
                log.warning(f"{tag}joint {_prim_name(prim)} ({_prim_path(prim)}): unknown {ATTR_MODEL} "
                            f"{str(model)!r} (expected \"bam\" or \"pd\") -> PhysX PD drive")
            specs.append(None)
            continue

        def reject(reason: str) -> None:
            log.warning(f"{tag}BAM servo model rejected on joint {_prim_name(prim)} ({_prim_path(prim)}) of "
                        f"{root_path}: {reason} -> this joint keeps the PhysX PD drive")
            specs.append(None)

        raw = _attr(prim, ATTR_PARAMS)
        if raw is None or not str(raw).strip():
            reject(f"{ATTR_MODEL} is \"bam\" but {ATTR_PARAMS} is missing or empty")
            continue
        try:
            params = json.loads(str(raw))
        except Exception as e:
            reject(f"{ATTR_PARAMS} is not valid JSON ({type(e).__name__}: {e})")
            continue
        if not isinstance(params, dict):
            reject(f"{ATTR_PARAMS} must be a JSON object, got {type(params).__name__}")
            continue
        merged = dict(defaults)
        merged.update(params)
        try:
            validate_params(merged)
        except ActuatorParamsError as e:
            reject(str(e))
            continue
        name = _attr(prim, ATTR_NAME)
        if name:
            merged["_name"] = str(name)
        specs.append(merged)
    return specs
