from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import omni.usd

from .. import config
from ..contact_forces import ContactForceBatch
from ..device.articulation import WrappedArticulation
from ..device.device import WrappedDevice
from ..device.rigid_object import WrappedRigidObject
from ..imu_batch import ImuBatch
from .interface import MessageHandler

if TYPE_CHECKING:
    from typing import Dict, List
    from isaacsim.core.api import SimulationContext

ARTICULATION = "Articulation"
RIGID_OBJECT = "RigidObject"


class Moderator:
    """Owns the command / state endpoint and the live devices, and runs the controller <-> Isaac Sim exchange once
    per physics step.

    Routing key = the prim name: an articulation's name is the robot_name of the gateways that drive it, a tracked
    object's name the topic of its object_state frames. One stage hosts any number of robots; each routes
    independently through this one endpoint.

    Per step (`synchronize`): decode the commands received since the last step, sample the stage-wide tensor views
    once, let every device apply its command and read its state, run the stage-wide IMU and contact-force passes,
    flush the batched drive targets, then publish every device's frames.
    """

    # Physics steps between retries of a device that failed to bind (half a second at 200 Hz).
    RETRY_EVERY_STEPS = 100
    # Physics steps without publishing after a robot is (re)bound: PhysX is still re-placing the articulation for a
    # few steps (joints in their pre-reset configuration, large velocities), and a controller must not act on that.
    MUTE_STEPS_AFTER_BIND = 40
    # Frame latch: physics steps without an app frame event after which commands are latched every step.
    LATCH_STALE_STEPS = 12
    # A device whose synchronize() raises gets no drive and no state update for that step; WARNING at most once per
    # device per period, with the number of errors in between.
    SYNC_ERROR_LOG_PERIOD_S = 10.0

    def __init__(self, sim_context: SimulationContext, host: str, subscribe_port: int, publish_port: int):
        self._logger = logging.getLogger("isaac_bridge")
        self._simulation_context = sim_context
        self._message_handler = MessageHandler(host=host, subscribe_port=subscribe_port, publish_port=publish_port)
        self._flag_running = False
        self._step_id = 0

        self._device_map: "Dict[str, Dict[str, dict]]" = {}    # kind -> name -> {"path": prim path}
        self._instance_map: "Dict[str, WrappedDevice]" = {}
        self._class_map = {ARTICULATION: WrappedArticulation, RIGID_OBJECT: WrappedRigidObject}
        self._retry_tick = 0              # physics steps until devices that failed to bind are tried again
        self._mute_steps = 0
        self._sync_errors = {}            # device name -> [time of the last WARNING, errors since]
        self._bind_attempted = set()      # names tried at least once since the last scan (later retries are quiet)
        self._unknown_targets = set()     # names that frames were addressed to without a device of that name

        # Stage-wide views shared with the devices (set by the extension; None = per-device reads)
        self._link_view = None
        self._batch = None
        self._imu_batch = None
        self._imu_batch_dirty = True
        self._imu_devices_seen = -1
        self._contact_batch = None
        self._contact_batch_dirty = True
        self._contact_devices_seen = -1

        # Frame latch (RC_ISAAC_FRAME_LATCH=1, see latch_frame)
        self._latch_enabled = config.frame_latch_enabled()
        self._frame_pending = False
        self._frames_seen = 0
        self._steps_in_frame = 0
        self._latch_logged = False
        self._latch_stale_logged = False

        self._init_device_map()

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> bool:
        """Bind the endpoint (once) and the devices of the current device map. False when the sockets could not be
        bound (another process holds the ports): the devices still run, nothing is published."""
        self._retry_tick = 0
        self._drop_stale_instances()
        # Timeline stop -> play with the devices kept: PhysX rebuilt its scene from USD, so every device re-applies
        # what it wrote through the tensor API (WrappedArticulation.on_resume), and PhysX re-places the robots.
        resumed = False
        for inst in self._instance_map.values():
            hook = getattr(inst, "on_resume", None)
            if hook:
                hook()
                resumed = True
        if resumed:
            self._mute_steps = self.MUTE_STEPS_AFTER_BIND
        self.create_device_instance()
        if not self._message_handler.is_server_running:
            self._message_handler.start_server()
        self._flag_running = True
        return bool(self._message_handler.is_server_running)

    def stop(self) -> None:
        self._message_handler.stop_server()
        self._reset_instance_map()
        self._flag_running = False

    # -- devices -----------------------------------------------------------------
    def _init_device_map(self) -> None:
        self._device_map = {ARTICULATION: {}, RIGID_OBJECT: {}}

    def update_device_map(self) -> None:
        """Find the robots (articulations) and tracked objects of the open stage."""
        self._init_device_map()
        self._bind_attempted.clear()
        self._unknown_targets.clear()
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        for prim in stage.Traverse():
            if WrappedArticulation.is_articulation(prim):
                kind = ARTICULATION
            elif WrappedRigidObject.is_tracked_object(prim):
                kind = RIGID_OBJECT
            else:
                continue
            name, path = str(prim.GetName()), str(prim.GetPath())
            self._logger.info(f"found {kind} {name} at {path}")
            for other in self._device_map.values():
                if name in other:
                    # the prim name is the routing key: two devices with one name cannot both be addressed
                    self._logger.warning(f"two devices are named '{name}' ({other[name]['path']} and {path}); "
                                         f"only {path} is served — give each robot a unique prim name")
                    other.pop(name)
                    break
            self._device_map[kind][name] = {"path": path}

    def create_device_instance(self) -> None:
        """Bind every device of the device map that is not bound yet. A device that cannot bind (its physics view
        is not valid yet) is retried from synchronize(); only the first attempt after a scan is logged."""
        for kind, items in self._device_map.items():
            for name, data in items.items():
                if name in self._instance_map:
                    continue
                path = data["path"]
                first_try = name not in self._bind_attempted
                self._bind_attempted.add(name)
                try:
                    instance = self._class_map[kind](sim_context=self._simulation_context, path=path, name=name)
                except Exception as e:
                    log = self._logger.error if first_try else self._logger.debug
                    log(f"could not bind {kind} {name}: {type(e).__name__}: {e}")
                    continue
                if not instance:
                    if first_try:
                        self._logger.info(f"{kind} {name}: physics not ready yet, retrying")
                    continue
                self._logger.info(f"bound {kind} {name} at {path}")
                self._instance_map[name] = instance
                self._attach_views(instance)
                self._imu_batch_dirty = True          # a new robot joins the stage-wide IMU pass
                self._contact_batch_dirty = True
                self._mute_steps = self.MUTE_STEPS_AFTER_BIND

    def _drop_stale_instances(self) -> None:
        """Destroy devices whose prim left the device map or moved to another path (the stage was edited between
        two plays); create_device_instance() then binds them afresh."""
        wanted = {name: data["path"] for items in self._device_map.values() for name, data in items.items()}
        for name in list(self._instance_map):
            inst = self._instance_map[name]
            if wanted.get(name) == getattr(inst, "_path", None):
                continue
            self._logger.info(f"{name} is no longer at {getattr(inst, '_path', '?')}; unbinding it")
            try:
                inst.destroy()
            except Exception as e:
                self._logger.debug(f"destroy {name}: {e}")
            del self._instance_map[name]
            self._imu_batch_dirty = True
            self._contact_batch_dirty = True

    def _reset_instance_map(self) -> None:
        for instance in self._instance_map.values():
            instance.destroy()
        self._instance_map = {}
        self._sync_errors = {}             # a new set of devices reports its first sync error right away
        self._invalidate_imu_batch()

    def _missing_devices(self) -> "List[str]":
        return [name for items in self._device_map.values() for name in items if name not in self._instance_map]

    def _articulations_ready(self) -> bool:
        """Every articulation of the device map has a live device (a valid PhysX handle — WrappedArticulation
        refuses to bind without one). The stage-wide articulation batch is only sampled once that holds: building a
        multi-articulation tensor view before PhysX has every robot is not safe."""
        wanted = list(self._device_map.get(ARTICULATION, {}))
        return bool(wanted) and all(name in self._instance_map for name in wanted)

    @property
    def device_map(self) -> "Dict[str, Dict[str, dict]]":
        return self._device_map

    @property
    def instances(self) -> "Dict[str, WrappedDevice]":
        return dict(self._instance_map)

    def articulation_paths(self) -> "List[str]":
        """Prim paths of every articulation in the device map (stage order)."""
        return [data["path"] for data in self._device_map.get(ARTICULATION, {}).values()]

    def tracked_paths(self) -> "List[str]":
        """Every rigid body the stage-wide link view tracks, stage order: the links of every articulation plus the
        free (non-kinematic) rigid bodies outside them. IMU links, foot-contact links, root links and tracked objects
        are all in this set, so their per-step poses come from one tensor read for the whole stage."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []
        from pxr import Usd, UsdPhysics
        try:
            from isaacsim.core.utils.articulations import find_all_articulation_base_paths
            robot_paths = [str(p) for p in find_all_articulation_base_paths()]
        except Exception as e:
            self._logger.warning(f"articulation base lookup failed ({e}); using the device map")
            robot_paths = self.articulation_paths()
        paths = []
        for robot_path in robot_paths:
            root = stage.GetPrimAtPath(robot_path)
            if not root or not root.IsValid():
                continue
            paths.extend(str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI))
        roots = tuple(p.rstrip("/") + "/" for p in robot_paths)
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            path = str(prim.GetPath())
            if path.startswith(roots) or path in robot_paths:
                continue
            kin = prim.GetAttribute("physics:kinematicEnabled")
            if kin and kin.IsValid() and bool(kin.Get()):
                continue
            paths.append(path)
        return list(dict.fromkeys(paths))

    # -- stage-wide views --------------------------------------------------------
    def set_link_view(self, view) -> None:
        """Share one batched rigid-body view with every device (and every device bound later)."""
        self._link_view = view
        self._invalidate_imu_batch()
        for inst in self._instance_map.values():
            self._attach_views(inst)

    def set_batch(self, batch) -> None:
        """Share the stage-wide articulation batch with every articulation device."""
        self._batch = batch
        for inst in self._instance_map.values():
            self._attach_views(inst)

    def _attach_views(self, inst) -> None:
        hook = getattr(inst, "set_link_view", None)
        if hook:
            hook(self._link_view)
        hook = getattr(inst, "set_batch", None)
        if hook:
            hook(self._batch)

    def _invalidate_imu_batch(self) -> None:
        """The device set or the link view changed: rebuild the IMU and contact-force passes at the next step."""
        if self._imu_batch is not None:
            for device in self._imu_batch.devices:
                hook = getattr(device, "set_imu_batched", None)
                if hook:
                    hook(False)
        self._imu_batch = None
        self._imu_batch_dirty = True
        self._imu_devices_seen = -1
        self._contact_batch = None
        self._contact_batch_dirty = True

    def _ensure_imu_batch(self):
        """One ImuBatch over every device that emulates an IMU, rebuilt only when that set changed (collecting the
        bindings allocates, so it must not run every step)."""
        if not self._imu_batch_dirty:
            return self._imu_batch
        self._imu_batch_dirty = False
        bindings = []
        for name in sorted(self._instance_map):
            device = self._instance_map[name]
            hook = getattr(device, "imu_binding", None)
            spec = hook() if hook else None
            if spec:
                bindings.append((device, spec))
        batch = ImuBatch(bindings, logger=self._logger) if bindings else None
        # Every device the new batch does NOT carry must be told: it kept the flag from the previous batch, and a
        # device that believes the stage-wide pass feeds it never samples its own IMU again.
        carried = {id(d) for d in (batch.devices if batch is not None else ())}
        for device in self._instance_map.values():
            if id(device) in carried:
                continue
            hook = getattr(device, "set_imu_batched", None)
            if hook:
                hook(False)
        self._imu_batch = batch
        return batch

    def _ensure_contact_batch(self):
        """One ContactForceBatch over every device with rc:contactForceLinks, rebuilt when that set changes."""
        if not self._contact_batch_dirty:
            return self._contact_batch
        self._contact_batch_dirty = False
        bindings = []
        for name in sorted(self._instance_map):
            device = self._instance_map[name]
            hook = getattr(device, "contact_force_binding", None)
            paths = hook() if hook else None
            if paths:
                bindings.append((device, paths))
        self._contact_batch = ContactForceBatch(bindings, logger=self._logger) if bindings else None
        return self._contact_batch

    # -- per app frame / physics step ---------------------------------------------
    def on_frame(self) -> None:
        """Kit app update event: the next physics step starts a new rendered frame."""
        self._frame_pending = True
        self._frames_seen += 1
        if self._flag_running and not self._simulation_context_playing():
            # stopped timeline: no physics step drains the inbound queue, so keep it from filling with stale frames
            # (a teleport sent while stopped is still applied at the next play)
            self._message_handler.drain_inbound(self._process_data)

    def _simulation_context_playing(self) -> bool:
        try:
            return bool(self._simulation_context.is_playing())
        except Exception:
            return True

    def synchronize(self, dt: float) -> None:
        if not self._flag_running:
            return
        self._step_id += 1
        # gateway commands received since the last step, decoded in arrival order
        self._message_handler.drain_inbound(self._process_data)
        if self._link_view is not None:
            self._link_view.sample(self._step_id)
        batch = self._batch if (self._batch is not None and self._articulations_ready()) else None
        if batch is not None:
            batch.sample(self._step_id)
        if self._latch_enabled:
            self._latch_frame()

        if self._retry_tick > 0:
            self._retry_tick -= 1
        elif self._missing_devices():
            # A device that could not bind (typically an articulation whose physics view was not valid yet) is
            # retried here, once the physics callback proves the simulation is stepping.
            self._retry_tick = self.RETRY_EVERY_STEPS
            self.create_device_instance()

        for name, instance in self._instance_map.items():
            try:
                instance.synchronize()
            except Exception as e:
                self._report_sync_error(name, e)

        # a robot reads its rc:imuLink on its first synchronize, so the set can grow after a bind
        n_imu = sum(1 for inst in self._instance_map.values() if getattr(inst, "has_imu", None) and inst.has_imu())
        if n_imu != self._imu_devices_seen:
            self._imu_devices_seen = n_imu
            self._imu_batch_dirty = True
        imu_batch = self._ensure_imu_batch()
        if imu_batch is not None:
            try:
                sim_time = float(self._simulation_context.current_time)
            except Exception:
                sim_time = 0.0
            batched = imu_batch.compute(self._link_view, sim_time)
            for device in imu_batch.devices:
                hook = getattr(device, "set_imu_batched", None)
                if hook:
                    hook(batched)
        n_forces = sum(1 for inst in self._instance_map.values()
                       if getattr(inst, "contact_force_binding", None) and inst.contact_force_binding())
        if n_forces != self._contact_devices_seen:
            self._contact_devices_seen = n_forces
            self._contact_batch_dirty = True
        contact_batch = self._ensure_contact_batch()
        if contact_batch is not None:
            try:
                contact_batch.compute(float(self._simulation_context.current_time), float(dt))
            except Exception as exc:
                self._logger.warning(f"contact forces pass failed: {type(exc).__name__}: {exc}")
        if batch is not None:
            batch.flush()          # one backend call per tensor the robots changed, for the whole stage

        if self._mute_steps > 0:
            self._mute_steps -= 1
            return
        for name, instance in self._instance_map.items():
            try:
                frames = instance.get_actual()
                if frames:
                    self._message_handler.send(name, frames)
            except Exception as e:
                self._logger.error(f"publish error for {name}: {e}")

    def _latch_frame(self) -> None:
        """Frame latch (RC_ISAAC_FRAME_LATCH=1): Isaac Sim runs the physics steps of a rendered frame back to back,
        so a command that arrives in the middle of them would switch the drive target between two steps of the same
        frame. At the first physics step of every frame each robot snapshots the command it holds for the whole
        frame (WrappedArticulation.latch_command). Until the first frame event arrives, or when frame events stop
        for more than LATCH_STALE_STEPS physics steps, commands are latched every step."""
        self._steps_in_frame += 1
        stale = self._steps_in_frame > self.LATCH_STALE_STEPS
        if stale and not self._latch_stale_logged:
            self._latch_stale_logged = True
            self._logger.warning(f"frame latch: {self._steps_in_frame} physics steps without an app frame event; "
                                 f"latching every step")
        if not (self._frame_pending or not self._frames_seen or stale):
            return
        if self._frame_pending and self._frames_seen and not self._latch_logged and self._steps_in_frame > 1:
            self._latch_logged = True
            self._logger.info(f"frame latch active: {self._steps_in_frame - 1} physics steps per app frame")
        self._frame_pending = False
        self._steps_in_frame = 0
        for inst in self._instance_map.values():
            latch = getattr(inst, "latch_command", None)
            if latch:
                latch()

    def _report_sync_error(self, name: str, exc: Exception, now: float = None) -> None:
        now = time.monotonic() if now is None else now
        state = self._sync_errors.get(name)
        if state is None:
            self._sync_errors[name] = [now, 0]
            self._logger.warning(f"sync error in {name}: {type(exc).__name__}: {exc} (the device gets no drive or "
                                 f"state update while this repeats; reported at most every "
                                 f"{self.SYNC_ERROR_LOG_PERIOD_S:g} s)", exc_info=exc)
            return
        state[1] += 1
        if now - state[0] >= self.SYNC_ERROR_LOG_PERIOD_S:
            self._logger.warning(f"sync error in {name}: {type(exc).__name__}: {exc} "
                                 f"({state[1]} error(s) in the last {now - state[0]:.0f} s)")
            state[0], state[1] = now, 0

    def _process_data(self, data: "List[bytes]") -> None:
        """One inbound multipart frame [b'TM/', name, data_type, payload] -> the device of that name."""
        try:
            if len(data) < 4:
                self._logger.debug(f"inbound frame with {len(data)} part(s) ignored")
                return
            name = data[1].decode()
            data_type = data[2].decode()
            device = self._instance_map.get(name)
            if device is not None:
                device.update_target(data_type, data[3])
            elif name not in self._unknown_targets:
                self._unknown_targets.add(name)
                if any(name in items for items in self._device_map.values()):
                    self._logger.info(f"'{data_type}' for {name}, which is not bound yet; dropped until it is")
                else:
                    robots = sorted(self._device_map.get(ARTICULATION, {}))
                    self._logger.warning(f"'{data_type}' for '{name}', which is no robot or tracked object of this "
                                         f"stage (robots: {robots or 'none'}); check the gateway's robot name")
        except Exception as e:
            self._logger.error(f"inbound frame: {type(e).__name__}: {e}")
