# Isaac Bridge

An Isaac Sim extension that makes Isaac Sim the simulator of a Sodero robot controller.

The bridge speaks the same wire as the Sodero cloud simulator. A controller's simulation gateways
(actuator, IMU and camera gateways) connect to an Isaac Sim running this extension exactly as they connect
to the cloud: the controller, its settings and its applications stay the same, only the simulator address
changes.

```
 Isaac Sim + Isaac Bridge                                   Sodero robot controller (Station)
 -------------------------------------------                ------------------------------------------
 articulations ----- sim_state, imu_state ...  -- :53334 --> actuator / IMU gateways --> controllers
              <----- sim_command               -- :53333 --- actuator gateways
 cameras ----------- camera_pair (RGB-D)       -- :53335 --> camera gateways --> VO, localizer, ...
```

Requirements: Isaac Sim 6.0 or later (Linux or Windows), and a network route from the controller to the
computer running Isaac Sim (TCP ports 53333-53335).

## Quick start

1. **Install.** In Isaac Sim, open *Window > Extensions*, open the settings (gear icon) and add the folder
   that *contains* the `isaac_bridge` folder to the extension search paths. Search for **Isaac Bridge**,
   enable it and tick *Autoload*. On first use Isaac Sim installs `pyzmq` from the Python package index;
   without internet access install it into Isaac Sim's Python yourself.

   From a command line instead:

   ```sh
   isaac-sim.sh --ext-folder /path/to/folder/containing/isaac_bridge --enable isaac_bridge
   ```

2. **Open a World and press Play.** The bridge finds every robot (articulation) and camera of the stage,
   binds its ports and starts serving. *Window > Isaac Bridge* shows what it serves.

3. **Point the controller at Isaac Sim.** On the Station page, enter the address of the computer running
   Isaac Sim in the gateways: **Sim host** in an actuator gateway's *TARGET* section, and **Camera host**
   (port 53335) in a camera gateway's settings. The IMU gateway has no address field: it connects to the
   simulator address configured for the controller, which Sodero support sets. The user manual describes
   every step: [Connect to Stellar Station](https://docs.sodero.io/latest/virtual_simulation/connect_station/).

4. **Name the gateways after the prims.** A gateway finds its robot by name:

   | Gateway | Robot name to enter | Example |
   | --- | --- | --- |
   | Actuator gateway (simulation) | the articulation prim name | `turtlebot3_burger` |
   | IMU gateway | the articulation prim name (the robot must author `rc:imuLink`) | `microduck` |
   | Camera gateway | the camera prim name | `cam0` |

   Enter the camera's field of view in the camera gateway's settings: the bridge logs it for every camera
   (`horizontal field of view 60.00 deg`).

Then start the nodes on the Station and switch the drives on: the robot moves in Isaac Sim.

## Worlds

Worlds exported by Sodero (and uploaded to the Sodero cloud) run unchanged. For other stages:

- **Robots.** Each robot is an articulation root (`UsdPhysics.ArticulationRootAPI` with
  `PhysxArticulationAPI`), and its prim name — unique in the stage — is its robot name. Actuator *i* is the
  *i*-th movable joint (revolute or prismatic) under the robot prim in USD order; joints marked
  `rc:passive = true` are left out. The bridge maps this order onto PhysX's DOF order itself.
- **Units.** Z up. Joint values cross the wire in radians (revolute) and metres (prismatic).
- **Rates.** The world's `PhysicsScene` prim may author `rc:physicsHz` and `rc:renderHz`; without them the
  bridge runs 60 Hz physics and 60 Hz rendering, as the Sodero simulator does. Isaac Sim is rate-limited
  to the render rate, so the simulation runs in real time while the computer keeps up.
- **Initial pose.** When a robot is bound (every Play) it is put back at its authored pose: the root link
  at its USD world pose, every joint at its `PhysicsJointStateAPI` position, velocities zero.

Optional attributes, on the robot (articulation root) prim unless noted. Link names refer to prims under
`<robot>/links/`:

| Attribute | Effect |
| --- | --- |
| `rc:positionKp`, `rc:positionKd` (one value per actuator) | POSITION-mode drive gains instead of the stiff defaults (small servo robots); with `rc:positionVelocityFeedforward = true` the drive also tracks the commanded velocity |
| `rc:positionDriveModel = "capped_p_passive"`, `rc:positionTorqueLimits`, `rc:positionFrictionLoss` (one value per actuator) | POSITION commands as an explicit torque: the P term saturates at the limit before damping and friction are deducted |
| `rc:imuLink`, `rc:imuOffset`, `rc:imuAccelTau` | publish `imu_state`: an IMU at that point of that link, its accelerometer low-passed with that time constant (default 0.02 s) |
| `rc:contactLinks`, `rc:contactOffsets`, `rc:contactHeight`, `rc:groundZ` | two foot-contact flags in `imu_state` |
| `rc:contactForceLinks` (up to 8 links) | publish `contact_forces`: the vertical contact load on each link |
| `rc:actuator:model = "bam"` and `rc:actuator:params` on joints, `rc:actuator:defaults` on the robot | servo actuator model instead of the PhysX drive ([`scripts/device/actuator_model.py`](../scripts/device/actuator_model.py)) |
| `rc:passive = true` on a joint, optionally `rc:backlashOf = "<joint name>"` | the joint is no actuator; a backlash hinge adds its angle to that joint's reading |

Tracked objects (a ball, a goal) carry `rc:trackedObject = true` and `rc:objectKind`; the bridge publishes
their pose as `object_state` and accepts `object_teleport` commands to move them.

## What the bridge serves

| Port | Direction | Topic `TM/<name>/<kind>` | When |
| --- | --- | --- | --- |
| 53333 | controller to bridge | `sim_command`, `object_teleport` | every control cycle |
| 53334 | bridge to controller | `sim_state` | every physics step, every robot |
| 53334 | bridge to controller | `imu_state`, `contact_forces` | every physics step, robots that ask for them |
| 53334 | bridge to controller | `root_pose` | 20 Hz of simulation time, every robot |
| 53334 | bridge to controller | `object_state` | 50 Hz (moving objects), 2 Hz (static ones) |
| 53335 | bridge to controller | `camera_pair`, `camera_info` | per capture / once a second, every camera |

Byte layouts are in [`scripts/protocol.py`](../scripts/protocol.py). The drive semantics are those of a
servo drive: POSITION, VELOCITY, TORQUE and IMPEDANCE per joint; a gateway switching its drive off sends
IMPEDANCE with zero stiffness (the joint goes limp); joints no gateway commands roll freely. Several
gateways can share one robot, each driving its own window of joints.

The simulation clock starts at zero on every Play. A gateway therefore sees a new simulator session after
Stop and Play, and keeps its drive off until it is switched on again — as after a real power cycle.

## Cameras

Every camera of the stage (each `UsdGeom.Camera` prim except Isaac Sim's own viewport cameras) is rendered
and published under its prim name, 640 x 480 at 15 frames per second: RGB and depth (distance to the image
plane, millimetres, 0 where nothing was hit), aligned and undistorted. Two cameras with the same prim name
are skipped with a warning. Cameras render only while the simulation plays.

Every frame carries its capture time on the computer's clock. The camera gateway accepts a frame captured
at most 20 ms in its future and 2 s in its past, so run time synchronisation (NTP) on both computers. If you
cannot, start Isaac Sim with `RC_CAMERA_CAPTURE_TIME=0`: the frames then carry no capture time and the
camera gateway stamps their arrival.

To change the size or rate of a camera, add a camera, or publish only listed cameras, set (before starting
Isaac Sim):

```sh
RC_CAMERA_AUTO_WIDTH=424 RC_CAMERA_AUTO_HEIGHT=240 RC_CAMERA_AUTO_FPS=30      # all automatic cameras
RC_ISAAC_CAMERA_DEVICES='[{"name": "cam0", "width": 848, "height": 480},
                          {"name": "rear", "prim_path": "/World/robot/base_link/rear_cam", "fps": 5}]'
RC_CAMERA_AUTO_PUBLISH=0                                                    # only the listed cameras
```

An entry may set `name`, `prim_path`, `width`, `height`, `fps`, `frame_id` (default `<name>_optical`, at
most 15 bytes) and `pub_port`. The same list can be written into `CAMERA_DEVICES` in
[`scripts/config.py`](../scripts/config.py).

## Settings

All settings are environment variables read when Isaac Sim starts; the defaults match the Sodero simulator.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RC_ISAAC_COMMAND_PORT` | 53333 | command port; the state port is this + 1 |
| `RC_CAMERA_PUB_PORT` | 53335 | camera port |
| `RC_ISAAC_PHYSICS_HZ`, `RC_ISAAC_RENDER_HZ` | the world's | override the world's rates |
| `RC_CAMERA_AUTO_PUBLISH`, `RC_CAMERA_AUTO_WIDTH`, `_HEIGHT`, `_FPS`, `RC_ISAAC_CAMERA_DEVICES` | on, 640, 480, 15, none | cameras (above) |
| `RC_CAMERA_CAPTURE_TIME` | 1 | 0: camera frames without capture time (above) |
| `RC_ISAAC_ROOT_POSE_HZ` | 20 | lower the `root_pose` rate; 0 turns it off |
| `RC_ISAAC_CONTACT_HEIGHT` | the world's | foot-contact threshold of every robot, metres |
| `RC_ISAAC_RESPAWN_AT_BIND` | 1 | 0: keep robots where they are on Play instead of their authored pose |
| `RC_ISAAC_FRAME_LATCH` | 0 | 1: hold each command for all physics steps of a rendered frame |

## Differences from the cloud simulator

- The cloud renders robot cameras in a separate renderer; here Isaac Sim renders them itself. A prim's
  `sodero:sensorVisibility` (a different look for sensors than for people) is not applied: the viewport and
  the cameras see the same stage.
- Pausing the timeline keeps the robots where they are; a gateway sees the simulator stop and keeps its
  drive off when it resumes. Stop and Play start a new session (above).
- Editing the stage while it plays is picked up on the next Play or with *Rescan* in the status window.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Status window: ports not bound | another program (a second Isaac Sim, a local simulator) holds ports 53333-53335 |
| Gateway: "waiting for simulator" | Isaac Sim is not playing, the address is wrong, or a firewall blocks the ports |
| Console: "'sim_command' for 'X', which is no robot or tracked object of this stage" | the gateway's robot name is not the prim name of an articulation of the stage |
| Robot found but never bound | it is not a PhysX articulation, or its prim name is used twice |
| Camera gateway: "Simulator ... publishes frames of camera(s) X but none of camera Y" | the camera gateway's robot name is not the camera prim name |
| Camera gateway: frames "stale or ... clock skew" | the two computers' clocks differ; synchronise them, or start Isaac Sim with `RC_CAMERA_CAPTURE_TIME=0` |
| Robot moves too fast or too slow | the computer cannot keep the render rate; lower camera sizes or rates, or close viewports |

The Isaac Sim console (and its log file) shows the bridge's messages, prefixed with the robot or camera name.
The bridge logs through the Python logger `isaac_bridge`; its DEBUG messages add, per robot, how joint state
and drive targets are exchanged with PhysX and what PhysX accepted of the first drive.

## Development

```sh
python -m pytest tests        # from this folder; needs numpy, pyzmq and pytest — no Isaac Sim
```

The tests replace Isaac Sim with fakes (`tests/conftest.py`, `tests/fakes.py`) and cover the wire layout,
command merging, drive resolution, servo models, IMU and contact emulation, root pose, tracked objects,
camera naming and packing.

With Isaac Sim, `tests/isaac/` has the integration tools:

| Tool | Use |
| --- | --- |
| `run_isaac.py WORLD.usd` | run Isaac Sim headless with the bridge on one world (Isaac Sim's Python) |
| `probe.py --robot NAME` | talk to a running bridge as the gateways do and check what comes back (numpy, pyzmq) |

| Path | Contents |
| --- | --- |
| `scripts/extension.py` | lifecycle: stage, timeline, rates, physics callback, cameras |
| `scripts/config.py` | ports, rates, default drive gains, settings |
| `scripts/protocol.py` | byte layouts of every message |
| `scripts/moderator/` | the ZMQ endpoint and the per-step exchange |
| `scripts/device/articulation.py` | one robot: commands to PhysX drives, state, IMU, root pose |
| `scripts/device/actuator_model.py` | servo actuator model (`rc:actuator:*`) |
| `scripts/device/rigid_object.py`, `root_pose.py` | tracked objects, root pose frames |
| `scripts/device/camera.py` | camera discovery, rendering and publishing |
| `scripts/articulation_batch.py`, `link_view.py`, `imu_batch.py`, `contact_forces.py` | stage-wide tensor reads and writes |
| `scripts/gui/panel.py` | the status window |
