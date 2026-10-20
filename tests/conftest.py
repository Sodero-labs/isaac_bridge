"""Stub the Isaac Sim / Kit / USD runtime so the extension's pure-Python logic imports and runs on a plain Python.

Only what the package's import chain touches at module level is stubbed; anything a test needs to behave (a
SimulationContext, a tensor view) is a small fake from tests/fakes.py or the test itself.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _module(name: str, **attrs) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install_stubs() -> None:
    if "omni" in sys.modules and getattr(sys.modules["omni"], "_isaac_bridge_stub", False):
        return
    omni = _module("omni")
    omni._isaac_bridge_stub = True
    _module("omni.usd", get_context=MagicMock(), StageEventType=MagicMock())
    _module("omni.ext", IExt=object)
    _module("omni.kit")
    _module("omni.kit.app", get_app=MagicMock())
    _module("omni.timeline", get_timeline_interface=MagicMock(), TimelineEventType=MagicMock())
    for parent, child in (("omni", "usd"), ("omni", "ext"), ("omni", "kit"), ("omni.kit", "app"), ("omni", "timeline")):
        setattr(sys.modules[parent], child, sys.modules[f"{parent}.{child}"])

    pxr = _module("pxr")
    for name in ("Usd", "UsdPhysics", "UsdGeom", "Gf", "Tf", "Sdf", "PhysxSchema"):
        setattr(pxr, name, MagicMock(name=f"pxr.{name}"))

    isaacsim = _module("isaacsim")
    core = _module("isaacsim.core")
    api = _module("isaacsim.core.api", SimulationContext=MagicMock(name="SimulationContext"))
    _module("isaacsim.core.api.simulation_context", SimulationContext=api.SimulationContext)
    utils = _module("isaacsim.core.utils")
    _module("isaacsim.core.utils.articulations", find_all_articulation_base_paths=MagicMock(return_value=[]))
    _module("isaacsim.core.utils.types", ArticulationActions=MagicMock(name="ArticulationActions"))
    _module("isaacsim.core.prims", Articulation=MagicMock(name="Articulation"), RigidPrim=MagicMock(name="RigidPrim"))
    _module("isaacsim.core.simulation_manager",
            SimulationManager=MagicMock(name="SimulationManager", get_physics_sim_view=MagicMock(return_value=None)))
    isaacsim.core = core
    core.api = api
    core.utils = utils
    core.prims = sys.modules["isaacsim.core.prims"]
    core.simulation_manager = sys.modules["isaacsim.core.simulation_manager"]
    utils.articulations = sys.modules["isaacsim.core.utils.articulations"]
    utils.types = sys.modules["isaacsim.core.utils.types"]


_install_stubs()
