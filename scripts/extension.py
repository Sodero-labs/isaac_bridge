"""Isaac Bridge — Isaac Sim as the simulator of a Sodero robot controller.

The extension speaks the wire of the Sodero cloud simulator, so a controller's simulation gateways (actuator,
IMU, camera) connect to this Isaac Sim exactly as they connect to the cloud:

  robots (moderator/, device/)   ports 53333 (commands in) / 53334 (state out), one exchange per physics step
  cameras (device/camera.py)     port 53335, rendered on Kit update events at their own rate

Lifecycle:
  stage opened   adopt (or create) the SimulationContext at the world's rates, find the robots
  Play           bind the ports (once per stage), bind the robots, start the cameras
  Stop           release the cameras; robots and ports stay (the next Play resumes them)
  stage closed   unbind everything
"""
from __future__ import annotations

import asyncio
import logging

import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd

from . import config
from .articulation_batch import ArticulationBatch
from .device.camera import CameraSet
from .link_view import LinkStateView
from .moderator.moderator import ARTICULATION, Moderator
from .sim_rates import world_rates

PHYSICS_CALLBACK_NAME = "isaac_bridge_physics_step"
_LOG = logging.getLogger("isaac_bridge")


class IsaacBridge(omni.ext.IExt):

    def on_startup(self, ext_id: str) -> None:
        _LOG.setLevel(logging.INFO)
        self._ext_id = ext_id
        self._ctx = None
        self._moderator = None
        self._cameras = CameraSet(_LOG)
        self._generation = 0
        self._reset_in_progress = False
        self._paused = False
        self._rates = (config.DEFAULT_PHYSICS_HZ, config.DEFAULT_RENDER_HZ)
        self._serving = None              # None = never started; True/False = ports bound
        self._last_error = ""
        self._link_view = None
        self._batch = None

        self._timeline = omni.timeline.get_timeline_interface()
        self._stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="isaac_bridge_stage")
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event, name="isaac_bridge_timeline")
        # Kit update = one rendered frame: the frame-latch tick of the moderator and the camera capture clock
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="isaac_bridge_update")

        self._panel = None
        try:
            from .gui.panel import PanelManager
            self._panel = PanelManager(ext_id, self)
        except Exception as e:  # noqa: BLE001 - no UI in a headless app
            _LOG.info(f"status window unavailable ({type(e).__name__}: {e})")

        _LOG.info(f"{config.EXT_NAME} started: commands on port {config.SUB_PORT}, state on {config.PUB_PORT}, "
                  f"cameras on {config.CAMERA_PUB_PORT}")
        if omni.usd.get_context().get_stage() is not None:
            asyncio.ensure_future(self._reset_project(self._generation))

    def on_shutdown(self) -> None:
        self._generation += 1                     # cancels a reset still in flight
        self._remove_physics_callback()
        self._cameras.stop()
        self._dispose_moderator()
        if self._panel is not None:
            self._panel.terminate()
            self._panel = None
        self._stage_sub = None
        self._timeline_sub = None
        self._update_sub = None
        _LOG.info(f"{config.EXT_NAME} stopped")

    # -- status (read by the panel) ------------------------------------------
    def status(self) -> dict:
        m = self._moderator
        robots = []
        if m is not None:
            bound = m.instances
            for kind, items in m.device_map.items():
                for name, data in items.items():
                    inst = bound.get(name)
                    robots.append({"name": name, "kind": kind, "path": data.get("path"), "bound": inst is not None,
                                   "dof": getattr(inst, "_n", None) if inst is not None else None})
        return {"ports": (config.SUB_PORT, config.PUB_PORT, config.CAMERA_PUB_PORT), "serving": self._serving,
                "error": self._last_error, "rates": self._rates, "playing": bool(self._timeline.is_playing()),
                "devices": robots, "cameras": self._cameras.status()}

    def rescan(self) -> None:
        """Find robots and cameras again (the stage was edited). Bound devices that still exist are kept."""
        if self._moderator is None:
            return
        self._moderator.update_device_map()
        if self._timeline.is_playing():
            self._on_play()

    # -- events ---------------------------------------------------------------
    def _on_stage_event(self, event) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            _LOG.info("stage opened: resetting the bridge")
            self._begin_reset()
            asyncio.ensure_future(self._reset_project(self._generation))
        elif event.type == int(omni.usd.StageEventType.CLOSED):
            self._begin_reset()

    def _begin_reset(self) -> None:
        self._generation += 1
        self._reset_in_progress = True
        self._paused = False
        self._remove_physics_callback()
        self._cameras.stop()
        self._dispose_moderator()

    def _on_timeline_event(self, event) -> None:
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if self._paused:
                # resumed from Pause: PhysX kept its scene, so the robots carry on where they were
                self._paused = False
                _LOG.info("resume")
            else:
                self._on_play()
        elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
            self._paused = True
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._paused = False
            self._on_stop()

    def _on_update(self, _event) -> None:
        m = self._moderator
        if m is not None:
            m.on_frame()
        if self._cameras.cameras and self._timeline.is_playing():
            self._cameras.on_update()
        if self._panel is not None:
            self._panel.tick()

    # -- play / stop ------------------------------------------------------------
    def _on_play(self) -> None:
        if self._reset_in_progress or self._moderator is None:
            return                                # _reset_project calls back once it is done
        stage = omni.usd.get_context().get_stage()
        self._enforce_rates()
        self._moderator.update_device_map()
        self._serving = self._moderator.start()
        if self._serving:
            self._last_error = ""
        else:
            self._last_error = (f"ports {config.SUB_PORT}/{config.PUB_PORT} could not be bound: another process "
                                f"(a second Isaac Sim, a local simulator) uses them")
            _LOG.error(self._last_error)
        self._bind_stage_views()
        self._cameras.start(stage)
        names = sorted(self._moderator.device_map.get(ARTICULATION, {}))
        _LOG.info(f"play: robots {names or 'none'}, cameras {[c.spec.name for c in self._cameras.cameras] or 'none'}")

    def _on_stop(self) -> None:
        self._cameras.stop()
        _LOG.info("stop: robots and ports kept for the next play")

    def _synchronize(self, dt: float) -> None:
        m = self._moderator
        if m is not None:
            m.synchronize(dt)

    # -- world (re)initialisation ---------------------------------------------------
    async def _reset_project(self, generation: int) -> None:
        app = omni.kit.app.get_app()
        # The SimulationContext clears itself on a stage open (its own stage-open callback): let that run first.
        await app.next_update_async()
        if generation != self._generation:
            return
        self._reset_in_progress = True
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            self._rates = world_rates(stage)
            ctx = await self._adopt_simulation_context(stage)
            if generation != self._generation or ctx is None:
                return
            self._ctx = ctx
            self._remove_physics_callback()
            ctx.add_physics_callback(PHYSICS_CALLBACK_NAME, self._synchronize)
            if self._moderator is None:
                self._moderator = Moderator(sim_context=ctx, host=config.BIND_HOST,
                                            subscribe_port=config.SUB_PORT, publish_port=config.PUB_PORT)
            self._moderator.update_device_map()
        except Exception as e:  # noqa: BLE001
            self._last_error = f"world setup failed: {type(e).__name__}: {e}"
            _LOG.exception(self._last_error)
            return
        finally:
            if generation == self._generation:
                self._reset_in_progress = False
        if self._timeline.is_playing():
            self._on_play()

    async def _adopt_simulation_context(self, stage):
        """The SimulationContext of this stage at the world's rates. One created by somebody else (a script, a
        World) is used as it is, with only its rates corrected; otherwise the bridge creates it, keeping the
        stage's units."""
        from isaacsim.core.api.simulation_context import SimulationContext
        from pxr import UsdGeom
        physics_hz, render_hz = self._rates
        ctx = SimulationContext.instance()
        if ctx is None:
            mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
            ctx = SimulationContext(physics_dt=1.0 / physics_hz, rendering_dt=1.0 / render_hz,
                                    stage_units_in_meters=mpu)
        if getattr(ctx, "_physics_context", None) is None:
            await ctx.initialize_simulation_context_async()
        self._ctx = ctx
        self._enforce_rates(log_always=True)
        return ctx

    def _enforce_rates(self, log_always: bool = False) -> None:
        """Keep PhysX stepping at the world's rate: a context re-initialised by somebody else falls back to its
        own (often 60 Hz) dt, and a 200 Hz world stepped at 1/60 s runs 3.3 times faster than real time."""
        ctx = self._ctx
        if ctx is None:
            return
        physics_hz, render_hz = self._rates
        try:
            current = (float(ctx.get_physics_dt()), float(ctx.get_rendering_dt()))
        except Exception:
            current = (0.0, 0.0)
        wanted = (1.0 / physics_hz, 1.0 / render_hz)
        changed = abs(current[0] - wanted[0]) > 1e-9 or abs(current[1] - wanted[1]) > 1e-9
        if changed:
            try:
                ctx.set_simulation_dt(physics_dt=wanted[0], rendering_dt=wanted[1])
            except Exception as e:  # noqa: BLE001
                _LOG.warning(f"could not set the simulation rates: {e}")
                return
        if changed or log_always:
            _LOG.info(f"rates: physics {physics_hz:g} Hz, render {render_hz:g} Hz "
                      f"({int(round(physics_hz / render_hz))} physics steps per frame)")

    def _bind_stage_views(self) -> None:
        """The two stage-wide tensor views every device shares: one RigidBodyView over every tracked link (IMU,
        foot contacts, root links, tracked objects) and one ArticulationView over every robot (joint state and
        drive targets). Both keep the per-step backend calls independent of the robot count."""
        m = self._moderator
        if m is None:
            return
        paths = m.tracked_paths()
        self._link_view = LinkStateView(paths, logger=_LOG) if paths else None
        m.set_link_view(self._link_view)
        robots = m.articulation_paths()
        self._batch = ArticulationBatch(robots, logger=_LOG) if robots else None
        m.set_batch(self._batch)

    def _remove_physics_callback(self) -> None:
        ctx = self._ctx
        try:
            if ctx is not None and ctx.physics_callback_exists(PHYSICS_CALLBACK_NAME):
                ctx.remove_physics_callback(PHYSICS_CALLBACK_NAME)
        except Exception as e:  # noqa: BLE001
            _LOG.debug(f"physics callback removal: {e}")

    def _dispose_moderator(self) -> None:
        m, self._moderator = self._moderator, None
        self._link_view = self._batch = None
        self._serving = None
        if m is not None:
            try:
                m.stop()
            except Exception as e:  # noqa: BLE001
                _LOG.warning(f"bridge stop: {type(e).__name__}: {e}")
