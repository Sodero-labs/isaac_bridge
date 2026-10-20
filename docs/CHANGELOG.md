# Changelog

## 1.0.0 — 2026-10-20

First release: Isaac Sim 6 as the simulator of a Sodero robot controller, on the wire of the Sodero cloud
simulator.

- **Robots.** Every articulation of the stage, addressed by its prim name: `sim_command` in, `sim_state` out
  every physics step. POSITION, VELOCITY, TORQUE and IMPEDANCE drives per joint; several gateways per robot,
  each on its own window of joints; uncommanded joints roll freely; per-joint teleports. Actuator order is
  the USD joint order, mapped onto PhysX's DOF order; passive and backlash joints (`rc:passive`,
  `rc:backlashOf`). Non-finite commands never reach PhysX.
- **Published effort.** A PD-driven joint publishes its drive torque, clamped to the drive's force limit;
  joints without a usable limit publish the measured joint torque.
- **Optional robot features** from USD attributes: POSITION drive gains, a capped-P position drive, an
  emulated IMU with foot contacts (`imu_state`), measured contact loads (`contact_forces`) and BAM servo
  actuator models (`rc:actuator:*`, validated per joint).
- **World state.** `root_pose` of every robot (20 Hz of simulation time) and `object_state` of tracked objects,
  which `object_teleport` moves.
- **Cameras.** Every camera of the stage rendered and published as RGB-D `camera_pair` frames with their
  capture time, plus `camera_info`; size, rate and selection configurable.
- **Lifecycle.** The SimulationContext of an opened stage is adopted at the world's rates (`rc:physicsHz`,
  `rc:renderHz`, default 60 / 60). On every Play robots are put back at their authored pose and gateways see
  a new simulator session; Pause resumes in place.
- **Performance.** One tensor view for every robot's joint state and drive targets, one for every tracked link,
  and one vectorized IMU pass: the per-step cost hardly grows with the robot count.
- Status window under *Window > Isaac Bridge*; a headless runner and a wire probe in `tests/isaac/`.
