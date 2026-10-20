"""Physics and render rates of a world, by the rule of the Sodero simulator.

physics: RC_ISAAC_PHYSICS_HZ > the PhysicsScene's `rc:physicsHz` > 60 Hz
render:  RC_ISAAC_RENDER_HZ > the PhysicsScene's `rc:renderHz` > 60 Hz, never above the physics rate, and
         lowered until every rendered frame runs a whole number of physics steps.

A 200 Hz world without rc:renderHz therefore renders at 50 Hz (4 physics steps per frame). Kit's run loop is
rate-limited to the render rate, so the simulation runs in real time when the computer keeps up.
"""
from __future__ import annotations

import math
import os

from . import config


def _env_hz(name: str, environ) -> "float | None":
    raw = str(environ.get(name, "") or "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def physics_scene_attr(stage, name: str) -> "float | None":
    """A positive number authored on the stage's first PhysicsScene prim, else None."""
    if stage is None:
        return None
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.HasAuthoredValue():
            try:
                value = float(attr.Get())
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) and value > 0 else None
        return None
    return None


def snap_render_hz(physics_hz: float, render_hz: float) -> float:
    render_hz = max(1.0, min(float(render_hz), float(physics_hz)))
    substeps = max(1, math.ceil(float(physics_hz) / render_hz - 1e-9))
    return float(physics_hz) / substeps


def world_rates(stage, environ=None) -> "tuple[float, float]":
    """(physics Hz, render Hz) for `stage` (see the module docstring)."""
    env = os.environ if environ is None else environ
    physics_hz = (_env_hz(config.PHYSICS_HZ_ENV, env) or physics_scene_attr(stage, "rc:physicsHz")
                  or config.DEFAULT_PHYSICS_HZ)
    render_hz = (_env_hz(config.RENDER_HZ_ENV, env) or physics_scene_attr(stage, "rc:renderHz")
                 or config.DEFAULT_RENDER_HZ)
    return float(physics_hz), snap_render_hz(physics_hz, render_hz)
