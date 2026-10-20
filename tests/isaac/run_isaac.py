"""Run the Isaac Bridge in a headless Isaac Sim on one world — the developer smoke test.

    <isaac-sim python> tests/isaac/run_isaac.py WORLD.usd [--seconds 60] [--no-render] [--stop-play 0]

Starts Isaac Sim without a window, enables the extension from this checkout, opens WORLD, presses Play and
steps the app in real time (one rendered frame per render period) for --seconds. Meanwhile anything can talk
to the bridge on ports 53333-53335 (tests/isaac/probe.py, a Station). --stop-play S presses Stop and Play
again after S seconds (a new simulator session). Prints the achieved real-time factor (simulated / wall time);
exits 2 when WORLD cannot be opened.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

EXT_FOLDER = str(Path(__file__).resolve().parents[3])        # the folder that contains isaac_bridge/


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world", help="USD stage to open")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run (wall time)")
    ap.add_argument("--no-render", action="store_true", help="no RTX renderer (no cameras)")
    ap.add_argument("--stop-play", type=float, default=0.0, help="press Stop and Play after this many seconds")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    extra = ["--ext-folder", EXT_FOLDER, "--enable", "isaac_bridge"]
    config = {"headless": True, "extra_args": extra}
    if args.no_render:
        # Storm instead of RTX: physics and the joint wire only, a fraction of the GPU (no cameras)
        extra += ["--/renderer/enabled=pxr", "--/renderer/active=pxr", "--/rtx/hydra/mdlMaterialWarmup=false",
                  "--/app/extensions/excluded=[omni.hydra.rtx,omni.hydra.rtx.shadercache.vulkan]"]
        config["create_new_stage"] = False      # no viewport handle to wait for
    app = SimulationApp(config)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    bridge_log = logging.getLogger("isaac_bridge")
    bridge_log.addHandler(handler)
    bridge_log.setLevel(logging.INFO)

    import omni.kit.app
    import omni.timeline
    import omni.usd

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaac_bridge"):
        manager.set_extension_enabled_immediate("isaac_bridge", True)
    print(f"[run_isaac] isaac_bridge enabled: {manager.is_extension_enabled('isaac_bridge')}", flush=True)

    result = omni.usd.get_context().open_stage(args.world)
    ok = result[0] if isinstance(result, tuple) else bool(result)      # Kit 110 returns a bool
    if not ok:
        print(f"[run_isaac] could not open {args.world}: {result}", flush=True)
        app.close()
        return 2
    for _ in range(30):                     # let the bridge adopt the stage (async reset)
        app.update()

    from isaacsim.core.api.simulation_context import SimulationContext
    ctx = SimulationContext.instance()
    render_dt = ctx.get_rendering_dt() if ctx is not None else 1.0 / 60.0
    print(f"[run_isaac] physics dt {ctx.get_physics_dt() if ctx else '?'} render dt {render_dt}", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    start = time.monotonic()
    session_start = start                   # the simulation clock starts again at every Play
    next_frame = start
    frames = 0
    replayed = args.stop_play <= 0
    while time.monotonic() - start < args.seconds:
        app.update()
        frames += 1
        if not replayed and time.monotonic() - start >= args.stop_play:
            replayed = True
            print("[run_isaac] Stop / Play", flush=True)
            timeline.stop()
            app.update()
            timeline.play()
            session_start = time.monotonic()
        next_frame += render_dt                 # real time: one rendered frame per render period
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_frame = time.monotonic()
    wall = time.monotonic() - start
    session = time.monotonic() - session_start
    sim = ctx.current_time if ctx is not None else 0.0
    print(f"[run_isaac] {frames} frames in {wall:.1f} s; since the last Play {sim:.2f} s of simulation in "
          f"{session:.1f} s (RTF {sim / max(session, 1e-6):.2f})", flush=True)
    timeline.stop()
    app.update()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
