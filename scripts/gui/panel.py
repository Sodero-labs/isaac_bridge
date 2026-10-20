from __future__ import annotations

import logging
import time
import weakref
from typing import TYPE_CHECKING

import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from isaacsim.gui.components.menu import make_menu_item_description

from ..config import EXT_NAME

if TYPE_CHECKING:
    from ..extension import IsaacBridge

REFRESH_PERIOD_S = 1.0
MENU = "Window"


class PanelManager:
    """Status window (Window > Isaac Bridge): ports, simulation rates, the robots, tracked objects and cameras
    the bridge serves. Everything is found automatically; Rescan picks up prims added while playing."""

    def __init__(self, ext_id: str, bridge: "IsaacBridge"):
        self._bridge = weakref.proxy(bridge)
        self._logger = logging.getLogger("isaac_bridge")
        self._last_refresh = 0.0
        self._window = ui.Window(EXT_NAME, width=460, height=380, visible=False)
        self._body = None
        with self._window.frame:
            with ui.VStack(spacing=4, height=0):
                self._body = ui.VStack(spacing=2, height=0)
                ui.Spacer(height=4)
                ui.Button("Rescan", height=24, clicked_fn=self._rescan)
        self._menu_items = [make_menu_item_description(ext_id, EXT_NAME, lambda p=weakref.proxy(self): p._toggle())]
        add_menu_items(self._menu_items, MENU)
        self._refresh()

    def _toggle(self) -> None:
        if self._window:
            self._window.visible = not self._window.visible
            if self._window.visible:
                self._refresh()

    def _rescan(self) -> None:
        try:
            self._bridge.rescan()
        except Exception as e:  # noqa: BLE001
            self._logger.warning(f"rescan failed: {e}")
        self._refresh()

    def tick(self) -> None:
        now = time.monotonic()
        if self._window is None or not self._window.visible or now - self._last_refresh < REFRESH_PERIOD_S:
            return
        self._refresh()

    def _refresh(self) -> None:
        self._last_refresh = time.monotonic()
        if self._body is None:
            return
        try:
            s = self._bridge.status()
        except Exception as e:  # noqa: BLE001
            s = {"error": str(e)}
        self._body.clear()
        with self._body:
            ports = s.get("ports") or ("-", "-", "-")
            if s.get("serving") is True:
                state = "serving"
            elif s.get("serving") is False:
                state = "not serving"
            else:
                state = "waiting for Play"
            ui.Label(f"Ports: commands {ports[0]}, state {ports[1]}, cameras {ports[2]} - {state}", height=18)
            if s.get("error"):
                ui.Label(s["error"], height=18, word_wrap=True, style={"color": 0xFF5050FF})
            rates = s.get("rates")
            if rates:
                ui.Label(f"Rates: physics {rates[0]:g} Hz, render {rates[1]:g} Hz", height=18)
            devices = s.get("devices") or []
            self._section("Robots", [d for d in devices if d.get("kind") == "Articulation"],
                          lambda d: f"{d['name']}   {d['path']}   "
                                    + (f"bound, {d['dof']} DOF" if d.get("bound") and d.get("dof")
                                       else "bound" if d.get("bound") else "not bound"))
            objects = [d for d in devices if d.get("kind") == "RigidObject"]
            if objects:
                self._section("Tracked objects", objects,
                              lambda d: f"{d['name']}   {d['path']}   {'bound' if d.get('bound') else 'not bound'}")
            self._section("Cameras", s.get("cameras") or [], self._camera_line)

    @staticmethod
    def _camera_line(c: dict) -> str:
        if c.get("error") and not c.get("running"):
            return f"{c['name']}   {c['error']}"
        size = c.get("size") or ("?", "?")
        return (f"{c['name']}   {size[0]}x{size[1]} at {c.get('fps', 0):g} fps   "
                f"{c.get('published', 0)} pairs sent")

    @staticmethod
    def _section(title: str, rows: list, fmt) -> None:
        ui.Spacer(height=4)
        ui.Label(title, height=18)
        if not rows:
            ui.Label("   none", height=18)
            return
        for row in rows:
            ui.Label("   " + fmt(row), height=18)

    def terminate(self) -> None:
        remove_menu_items(self._menu_items, MENU)
        if self._window is not None:
            self._window.destroy()
        self._window = None
        self._body = None
