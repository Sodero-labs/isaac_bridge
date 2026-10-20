"""Static configuration of the Isaac Bridge extension.

Everything here can be left alone: the defaults are the ports and behaviour of the Sodero cloud simulator,
which is what a robot controller expects. Environment variables override them without editing the
installed extension (set them before Isaac Sim starts); docs/README.md lists them all.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

EXT_NAME = "Isaac Bridge"


def env_flag(name: str, default: bool, environ=None) -> bool:
    """An on/off environment variable: 1/true/yes/on is on, any other value off, unset or empty the default."""
    raw = str((os.environ if environ is None else environ).get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _port(env: str, default: int) -> int:
    raw = os.environ.get(env, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if 1 <= value <= 65534 else default


# ---------------------------------------------------------------------------
# Network. The bridge BINDS these ports on every interface; the controller's gateways CONNECT to them.
#   gateway PUB -> bridge SUB  COMMAND_PORT       sim_command, object_teleport
#   bridge PUB  -> gateway SUB COMMAND_PORT + 1   sim_state, imu_state, contact_forces, root_pose, object_state
#   bridge PUB  -> camera SUB  CAMERA_PUB_PORT    camera_pair, camera_info
# The actuator gateway always uses command port + 1 for the state port.
# ---------------------------------------------------------------------------
BIND_HOST = "0.0.0.0"
COMMAND_PORT_ENV = "RC_ISAAC_COMMAND_PORT"
CAMERA_PUB_PORT_ENV = "RC_CAMERA_PUB_PORT"

SUB_PORT = _port(COMMAND_PORT_ENV, 53333)
PUB_PORT = SUB_PORT + 1
CAMERA_PUB_PORT = _port(CAMERA_PUB_PORT_ENV, 53335)

# ---------------------------------------------------------------------------
# Simulation rates. A world states the rates it was built for on its PhysicsScene prim
# (`rc:physicsHz`, `rc:renderHz`); without them the bridge runs 60 Hz physics and 60 Hz rendering, as the
# Sodero simulator does. The environment variables override the world. The render rate is snapped so that
# every rendered frame runs a whole number of physics steps.
# ---------------------------------------------------------------------------
PHYSICS_HZ_ENV = "RC_ISAAC_PHYSICS_HZ"
RENDER_HZ_ENV = "RC_ISAAC_RENDER_HZ"
DEFAULT_PHYSICS_HZ = 60.0
DEFAULT_RENDER_HZ = 60.0

# ---------------------------------------------------------------------------
# Joint drives (per DOF). POSITION / VELOCITY / TORQUE use these gains; IMPEDANCE uses the kp/kd carried in
# the command. The actuator gateway sends kp = kd = 0 for the non-impedance modes, so the simulator supplies
# the servo stiffness itself — otherwise a POSITION command would not track and an arm would collapse under
# gravity. A robot that authors `rc:positionKp` / `rc:positionKd` on its articulation root uses those
# POSITION gains instead (small servo-driven robots).
# ---------------------------------------------------------------------------
POSITION_KP = 1.0e7
POSITION_KD = 1.0e6
VELOCITY_KD = 1.0e4
TORQUE_KD = 0.0  # pure effort

# Damping of articulation DOFs no gateway commands (passive casters, idle wheels): kp = 0 (never
# position-locked) plus this light velocity damping, which mimics bearing friction and tames swivel shimmy
# without adding drive drag. 0.0 = fully free.
IDLE_KD = 1.0

# ---------------------------------------------------------------------------
# Robots.
#   RC_ISAAC_RESPAWN_AT_BIND=0   keep robots where they are on Play instead of putting them back at their
#                                authored pose
#   RC_ISAAC_FRAME_LATCH=1       hold each command for all physics steps of a rendered frame
#   RC_ISAAC_CONTACT_HEIGHT      foot-contact threshold of every robot [m], instead of rc:contactHeight
#   RC_ISAAC_ROOT_POSE_HZ        root_pose rate (at most 20 Hz of simulation time); 0 turns it off
# ---------------------------------------------------------------------------
RESPAWN_AT_BIND_ENV = "RC_ISAAC_RESPAWN_AT_BIND"
FRAME_LATCH_ENV = "RC_ISAAC_FRAME_LATCH"
CONTACT_HEIGHT_ENV = "RC_ISAAC_CONTACT_HEIGHT"
ROOT_POSE_HZ_ENV = "RC_ISAAC_ROOT_POSE_HZ"


def respawn_at_bind(environ=None) -> bool:
    return env_flag(RESPAWN_AT_BIND_ENV, True, environ)


def frame_latch_enabled(environ=None) -> bool:
    return env_flag(FRAME_LATCH_ENV, False, environ)


def contact_height_override(environ=None) -> Optional[float]:
    """RC_ISAAC_CONTACT_HEIGHT in metres, or None when unset; an invalid value is reported and ignored."""
    raw = str((os.environ if environ is None else environ).get(CONTACT_HEIGHT_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0.0:
        logging.getLogger("isaac_bridge").error(f"{CONTACT_HEIGHT_ENV}={raw!r} ignored: expected metres >= 0")
        return None
    return value


# ---------------------------------------------------------------------------
# Cameras. By default every camera of the stage (each UsdGeom.Camera prim except Kit's own viewport
# cameras) is rendered and published under its prim name, at AUTO width x height and fps — the rule of the
# Sodero cloud's camera renderer. A camera gateway subscribes with robot_name = that prim name.
#
#   RC_CAMERA_AUTO_PUBLISH=0                 publish only the cameras listed explicitly (below)
#   RC_CAMERA_AUTO_WIDTH / _HEIGHT / _FPS    size and rate of the automatic cameras (640 x 480 at 15)
#   RC_ISAAC_CAMERA_DEVICES                  explicit cameras, a JSON list of the CAMERA_DEVICES entries;
#                                            an entry wins over the automatic camera of the same name
#   RC_CAMERA_CAPTURE_TIME=0                 frames without their capture time: the camera gateway stamps
#                                            their arrival instead (for computers whose clocks are not
#                                            synchronized; the gateway rejects capture times more than 20 ms
#                                            ahead of or 2 s behind its own clock)
#
# CAMERA_DEVICES entry keys:
#   name       topic name (TM/<name>/camera_pair) = the camera gateway's robot_name; no '/' or '|'
#   prim_path  the UsdGeom.Camera prim
#   width, height, fps (default 640, 480, 15); frame_id (default <name>_optical, at most 15 bytes);
#   pub_port (default CAMERA_PUB_PORT)
# ---------------------------------------------------------------------------
CAMERA_CAPTURE_TIME_ENV = "RC_CAMERA_CAPTURE_TIME"
CAMERA_AUTO_ENV = "RC_CAMERA_AUTO_PUBLISH"
CAMERA_AUTO_WIDTH_ENV = "RC_CAMERA_AUTO_WIDTH"
CAMERA_AUTO_HEIGHT_ENV = "RC_CAMERA_AUTO_HEIGHT"
CAMERA_AUTO_FPS_ENV = "RC_CAMERA_AUTO_FPS"
CAMERA_DEVICES_ENV = "RC_ISAAC_CAMERA_DEVICES"
CAMERA_AUTO_DEFAULT = True
CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480
CAMERA_DEFAULT_FPS = 15.0
CAMERA_DEVICES = [
    # {"name": "front_cam", "prim_path": "/World/robot/base_link/front_cam",
    #  "width": 640, "height": 480, "fps": 15.0, "frame_id": "front_cam_optical"},
]
