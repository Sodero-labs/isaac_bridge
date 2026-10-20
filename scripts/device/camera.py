"""Robot cameras rendered by Isaac Sim and published to the controller's camera gateways (their `isaac` input).

Naming and wire are those of the Sodero cloud's camera renderer, so a camera gateway configured for the cloud
works here unchanged:

* Every camera of the stage — each UsdGeom.Camera prim except Kit's viewport cameras (/OmniverseKit_*) and
  replicator's (/Render/*) — is published under its prim name: topic ``TM/<prim name>/camera_pair``,
  frame_id ``<prim name>_optical``, 640 x 480 at 15 fps (RC_CAMERA_AUTO_* in config.py). Two cameras with
  the same prim name are ambiguous and skipped unless listed explicitly.
* Explicit entries (config.CAMERA_DEVICES, RC_ISAAC_CAMERA_DEVICES) add cameras or override the size, rate,
  frame_id or port of the automatic camera of the same name. RC_CAMERA_AUTO_PUBLISH=0 publishes only those.

Data path, per camera::

    Kit update ──(wall-clock fps gate, timeline playing)──▶ replicator annotators of one render product
        rgb (H, W, 4 uint8) + distance_to_image_plane (H, W float32, stage units)
        ──▶ rgb8, depth → uint16 mm (no hit / far clip → 0)
        ──▶ [FrameHdr depth][depth16][FrameHdr rgb][rgb8], FLAG_CAPTURE_UNIX_TIME + time.time_ns()
        ──▶ ZMQ PUB tcp://*:53335   b'TM/<name>/camera_pair||<blob>'   (SNDHWM 2: a slow gateway drops pairs)
        plus b'TM/<name>/camera_info||<json>' once a second (intrinsics from the USD camera)

Capture runs on Kit update events, never in the physics callback: the joint exchange with the actuator
gateways must not wait for a render. Depth is ``distance_to_image_plane`` (z depth, what a pinhole
back-projection expects), as in the cloud renderer. Intrinsics: square pixels,
fx = fy = focalLength * width / horizontalAperture, principal point at the image centre — the camera gateway
holds its own copy of these (enter the logged field of view there); camera_info is informational.

Everything Kit/Omniverse is imported inside the methods that need it, so the pure parts (configuration,
naming, conversion, packing) import and test on a plain Python with numpy (tests/test_camera_packing.py).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import config
from .. import protocol

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

_LOG = logging.getLogger("isaac_bridge")

RGB_ANNOTATOR = "rgb"
DEPTH_ANNOTATOR = "distance_to_image_plane"
CAMERA_INFO_PERIOD_S = 1.0
STATS_PERIOD_S = 10.0
EXCLUDED_PREFIXES = ("/OmniverseKit_", "/Render/")
FRAME_ID_SUFFIX = "_optical"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    width: int = config.CAMERA_DEFAULT_WIDTH
    height: int = config.CAMERA_DEFAULT_HEIGHT
    fps: float = config.CAMERA_DEFAULT_FPS
    frame_id: str = ""
    pub_port: int = config.CAMERA_PUB_PORT

    @classmethod
    def from_dict(cls, d: "Dict[str, Any]", default_port: "Optional[int]" = None) -> "CameraSpec":
        try:
            name = str(d["name"])
            prim_path = str(d["prim_path"])
        except KeyError as e:
            raise ValueError(f"camera entry is missing {e}: {d}")
        if not name or "/" in name or "|" in name:
            raise ValueError(f"camera name {name!r} is part of the ZMQ topic; no '/' or '|' allowed")
        spec = cls(
            name=name, prim_path=prim_path,
            width=int(d.get("width", config.CAMERA_DEFAULT_WIDTH)),
            height=int(d.get("height", config.CAMERA_DEFAULT_HEIGHT)),
            fps=float(d.get("fps", config.CAMERA_DEFAULT_FPS)),
            frame_id=str(d.get("frame_id") or default_frame_id(name)),
            pub_port=int(d.get("pub_port", config.CAMERA_PUB_PORT if default_port is None else default_port)),
        )
        if not prim_path.startswith("/"):
            raise ValueError(f"camera {name}: prim_path {prim_path!r} must be absolute")
        if spec.width <= 0 or spec.height <= 0:
            raise ValueError(f"camera {name}: resolution {spec.width}x{spec.height} is invalid")
        if not math.isfinite(spec.fps) or spec.fps <= 0:
            raise ValueError(f"camera {name}: fps {spec.fps} must be > 0")
        if len(spec.frame_id.encode("utf-8")) > protocol.FRAME_ID_MAX_LEN:
            raise ValueError(f"camera {name}: frame_id {spec.frame_id!r} is longer than "
                             f"{protocol.FRAME_ID_MAX_LEN} bytes")
        if not 1 <= spec.pub_port <= 65535:
            raise ValueError(f"camera {name}: pub_port {spec.pub_port} is not a port")
        return spec

    def with_overrides(self, d: "Dict[str, Any]") -> "CameraSpec":
        merged = {"name": self.name, "prim_path": self.prim_path, "width": self.width, "height": self.height,
                  "fps": self.fps, "frame_id": self.frame_id, "pub_port": self.pub_port}
        merged.update({k: v for k, v in d.items() if v is not None})
        return CameraSpec.from_dict(merged)


def topic_safe_name(name: str) -> str:
    """A prim name as a topic name: anything that would break ``TM/<name>/...`` becomes '_'."""
    out = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(name).strip())
    return out or "camera"


def default_frame_id(name: str) -> str:
    """``<name>_optical`` cut to the 15-byte header field (the name part is shortened, the suffix kept)."""
    budget = protocol.FRAME_ID_MAX_LEN - len(FRAME_ID_SUFFIX)
    stem = topic_safe_name(name).encode("utf-8")[:budget].decode("utf-8", errors="ignore") or "cam"
    return stem + FRAME_ID_SUFFIX


def auto_publish_enabled(environ=None) -> bool:
    return config.env_flag(config.CAMERA_AUTO_ENV, config.CAMERA_AUTO_DEFAULT, environ)


def capture_time_enabled(environ=None) -> bool:
    return config.env_flag(config.CAMERA_CAPTURE_TIME_ENV, True, environ)


def auto_defaults(environ=None, logger: "Optional[logging.Logger]" = None) -> "Dict[str, Any]":
    """Size, rate and port of the automatic cameras (bad values: the default, with an error log)."""
    env = os.environ if environ is None else environ
    log = logger or _LOG
    out: "Dict[str, Any]" = {"width": config.CAMERA_DEFAULT_WIDTH, "height": config.CAMERA_DEFAULT_HEIGHT,
                             "fps": config.CAMERA_DEFAULT_FPS, "pub_port": config.CAMERA_PUB_PORT}
    for key, name, cast in (("width", config.CAMERA_AUTO_WIDTH_ENV, int),
                            ("height", config.CAMERA_AUTO_HEIGHT_ENV, int),
                            ("fps", config.CAMERA_AUTO_FPS_ENV, float)):
        raw = str(env.get(name, "")).strip()
        if not raw:
            continue
        try:
            value = cast(raw)
            if not value > 0:
                raise ValueError("must be > 0")
            out[key] = value
        except ValueError as e:
            log.error(f"{name}={raw!r} ignored ({e}); using {out[key]}")
    return out


def load_camera_specs(config_list: "Optional[List[Dict[str, Any]]]" = None, env_value: "Optional[str]" = None,
                      default_port: "Optional[int]" = None,
                      logger: "Optional[logging.Logger]" = None) -> "List[Dict[str, Any]]":
    """The explicit camera entries: config.CAMERA_DEVICES, then RC_ISAAC_CAMERA_DEVICES (a JSON list; an entry
    with the same name replaces the earlier one). Returned as validated dicts — an entry may name only what it
    overrides of an automatic camera (name + width, say), so the prim path is checked when it is resolved.
    A bad entry is logged and skipped: a typo in one camera must never take the joint path down."""
    log = logger or _LOG
    raw: "List[Any]" = list(config.CAMERA_DEVICES if config_list is None else config_list)
    if env_value is None:
        env_value = os.environ.get(config.CAMERA_DEVICES_ENV, "")
    if env_value and env_value.strip():
        try:
            extra = json.loads(env_value)
            if not isinstance(extra, list):
                raise ValueError("expected a JSON list")
            raw.extend(extra)
        except Exception as e:  # noqa: BLE001 - configuration error, report and continue
            log.error(f"{config.CAMERA_DEVICES_ENV} ignored: {e}")
    entries: "Dict[str, Dict[str, Any]]" = {}
    for d in raw:
        try:
            if not isinstance(d, dict) or not d.get("name"):
                raise ValueError(f"camera entry needs a name: {d}")
            entry = dict(d)
            if default_port is not None and "pub_port" not in entry:
                entry["pub_port"] = default_port
            probe = dict(entry)
            probe.setdefault("prim_path", "/probe")        # validate everything except a missing path
            CameraSpec.from_dict(probe)
        except Exception as e:  # noqa: BLE001
            log.error(f"camera entry skipped: {e}")
            continue
        entries[str(entry["name"])] = entry
    return list(entries.values())


def discover_cameras(stage, logger: "Optional[logging.Logger]" = None) -> "List[Tuple[str, str]]":
    """[(topic name, prim path)] of every robot camera of the stage, stage order. A prim name used by two
    cameras is ambiguous: both are skipped with a warning (an explicit entry can still publish either)."""
    from pxr import UsdGeom
    log = logger or _LOG
    found: "Dict[str, List[str]]" = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        found.setdefault(topic_safe_name(prim.GetName()), []).append(path)
    out = []
    for name, paths in found.items():
        if len(paths) == 1:
            out.append((name, paths[0]))
        else:
            log.warning(f"[CAMERA] several cameras are named '{name}' ({', '.join(paths)}); none of them is "
                        f"published automatically — list the one to publish in {config.CAMERA_DEVICES_ENV}")
    return out


def resolve_specs(discovered: "List[Tuple[str, str]]", explicit: "List[Dict[str, Any]]", *, auto: bool,
                  defaults: "Dict[str, Any]", logger: "Optional[logging.Logger]" = None) -> "List[CameraSpec]":
    """The cameras to run: the automatic ones (when `auto`) at `defaults`, each explicit entry overriding the
    automatic camera of its name or adding a camera of its own."""
    log = logger or _LOG
    specs: "Dict[str, CameraSpec]" = {}
    if auto:
        for name, path in discovered:
            specs[name] = CameraSpec.from_dict({"name": name, "prim_path": path, **defaults})
    for entry in explicit:
        name = str(entry["name"])
        try:
            if name in specs:
                specs[name] = specs[name].with_overrides(entry)
            elif entry.get("prim_path"):
                specs[name] = CameraSpec.from_dict(entry)
            else:
                log.error(f"[CAMERA] {name}: no camera of that name in the stage and no prim_path given")
        except Exception as e:  # noqa: BLE001
            log.error(f"[CAMERA] {name}: entry skipped: {e}")
    return list(specs.values())


# ---------------------------------------------------------------------------
# Pure conversion helpers (no Kit)
# ---------------------------------------------------------------------------
def intrinsics_from_camera(focal_length_mm: float, horizontal_aperture_mm: float,
                           width: int, height: int) -> "Dict[str, float]":
    """Pinhole intrinsics of a USD camera as Kit renders it: the vertical field of view follows from the
    horizontal aperture and the render aspect, so pixels are square (fy == fx) and the principal point is the
    image centre."""
    fx = float(focal_length_mm) * float(width) / float(horizontal_aperture_mm)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}


def horizontal_fov_deg(focal_length_mm: float, horizontal_aperture_mm: float) -> float:
    return math.degrees(2.0 * math.atan(float(horizontal_aperture_mm) / (2.0 * float(focal_length_mm))))


def camera_info_json(name: str, frame_id: str, width: int, height: int,
                     intrinsics: "Dict[str, float]", fps: float) -> bytes:
    """The camera_info side channel, in the cloud renderer's format (compact: a few hundred bytes)."""
    return json.dumps({
        "name": name, "frame_id": frame_id, "width": int(width), "height": int(height), "fps": float(fps),
        "intrinsics": {"fx": round(float(intrinsics["fx"]), 4), "fy": round(float(intrinsics["fy"]), 4),
                       "cx": round(float(intrinsics["cx"]), 4), "cy": round(float(intrinsics["cy"]), 4)},
        "distortion_model": "none",
    }, separators=(",", ":")).encode("utf-8")


def rgba_to_rgb8(rgba: np.ndarray) -> np.ndarray:
    """(H, W, 3|4) colour buffer -> (H, W, 3) uint8, C-contiguous."""
    px = np.asarray(rgba)
    if px.ndim != 3 or px.shape[2] < 3:
        raise ValueError(f"colour buffer has shape {px.shape}, expected (H, W, 3|4)")
    if px.dtype != np.uint8:
        px = np.clip(px, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(px[:, :, :3])


def depth_to_mm(depth: np.ndarray, meters_per_unit: float) -> np.ndarray:
    """(H, W[, 1]) float depth in stage units -> (H, W) uint16 millimetres; no hit / far clip -> 0."""
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]
    if d.ndim != 2:
        raise ValueError(f"depth buffer has shape {d.shape}, expected (H, W)")
    return protocol.depth_m_to_mm(d.astype(np.float32) * float(meters_per_unit))


class FpsGate:
    """Decimate an update stream to `fps` on a fixed grid (no drift; a stall does not burst afterwards)."""

    def __init__(self, fps: float):
        self.period = 1.0 / float(fps)
        self._next = None

    def due(self, now: float) -> bool:
        if self._next is None:
            self._next = now + self.period
            return True
        if now < self._next:
            return False
        self._next += self.period * max(1, int((now - self._next) / self.period) + 1)
        return True


# ---------------------------------------------------------------------------
# Shared PUB socket (one bind per port, any number of cameras)
# ---------------------------------------------------------------------------
class _PairPublisher:
    """Reference-counted ZMQ PUB per port; cameras on one port are told apart by their topic prefix."""

    _by_port: "Dict[int, _PairPublisher]" = {}
    _guard = threading.Lock()

    def __init__(self, port: int):
        import zmq
        self.port = port
        self._refs = 0
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 2)     # at most two pairs queued per subscriber: latest wins
        self._sock.setsockopt(zmq.LINGER, 0)
        try:
            self._sock.bind(f"tcp://{config.BIND_HOST}:{port}")
        except Exception:
            self._sock.close(linger=0)
            self._ctx.term()
            raise

    @classmethod
    def acquire(cls, port: int) -> "_PairPublisher":
        with cls._guard:
            pub = cls._by_port.get(port)
            if pub is None:
                pub = cls(port)
                cls._by_port[port] = pub
            pub._refs += 1
            return pub

    def release(self) -> None:
        with self._guard:
            self._refs -= 1
            if self._refs > 0:
                return
            self._by_port.pop(self.port, None)
        try:
            self._sock.close(linger=0)
            self._ctx.term()
        except Exception as e:  # noqa: BLE001
            _LOG.warning(f"[CAMERA] PUB :{self.port} cleanup error: {e}")

    def send(self, topic: bytes, payload: bytes) -> bool:
        """Non-blocking ``topic||payload``; False when dropped at the HWM."""
        import zmq
        try:
            self._sock.send(topic + b"||" + payload, flags=zmq.NOBLOCK, copy=False)
            return True
        except zmq.Again:
            return False


# ---------------------------------------------------------------------------
# One camera: one render product, two annotators, one topic
# ---------------------------------------------------------------------------
class RcCamera:
    def __init__(self, spec: CameraSpec, logger: "Optional[logging.Logger]" = None, capture_time: bool = True):
        self.spec = spec
        self.log = logger or _LOG
        self._capture_time = bool(capture_time)
        self._pub: "Optional[_PairPublisher]" = None
        self._rp = None
        self._annot_rgb = None
        self._annot_depth = None
        self._gate = FpsGate(spec.fps)
        self._topic_pair = f"TM/{spec.name}/{protocol.CAMERA_PAIR_TOPIC}".encode()
        self._topic_info = f"TM/{spec.name}/{protocol.CAMERA_INFO_TOPIC}".encode()
        self._focal_mm, self._aperture_mm = 18.147562, 20.955   # Kit's default camera until the prim is read
        self._intrinsics = intrinsics_from_camera(self._focal_mm, self._aperture_mm, spec.width, spec.height)
        self._meters_per_unit = 1.0
        self._pair_seq = 0
        self._published = 0
        self._dropped = 0
        self._failed = 0
        self._size_warned = False
        self._last_info_at = 0.0
        self._last_stats_at = 0.0
        self._last_published_at = 0.0
        self.error: "Optional[str]" = None

    @property
    def running(self) -> bool:
        return self._rp is not None

    # -- lifecycle ----------------------------------------------------------
    def start(self, stage) -> bool:
        from pxr import UsdGeom
        prim = stage.GetPrimAtPath(self.spec.prim_path) if stage is not None else None
        if not prim or not prim.IsValid():
            self.error = f"no prim at {self.spec.prim_path}"
            self.log.warning(f"[CAMERA] {self.spec.name}: {self.error}")
            return False
        if not prim.IsA(UsdGeom.Camera):
            self.error = f"{self.spec.prim_path} is a {prim.GetTypeName()}, not a Camera"
            self.log.error(f"[CAMERA] {self.spec.name}: {self.error}")
            return False
        self._read_intrinsics(prim)
        self._meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
        try:
            self._pub = _PairPublisher.acquire(self.spec.pub_port)
        except Exception as e:  # noqa: BLE001
            self.error = f"cannot bind port {self.spec.pub_port}: {e}"
            self.log.error(f"[CAMERA] {self.spec.name}: {self.error}")
            return False
        try:
            import omni.replicator.core as rep
            self._rp = rep.create.render_product(self.spec.prim_path, (self.spec.width, self.spec.height))
            self._annot_rgb = rep.AnnotatorRegistry.get_annotator(RGB_ANNOTATOR)
            self._annot_depth = rep.AnnotatorRegistry.get_annotator(DEPTH_ANNOTATOR)
            self._annot_rgb.attach([self._rp])
            self._annot_depth.attach([self._rp])
        except Exception as e:  # noqa: BLE001
            self.error = f"render product: {type(e).__name__}: {e}"
            self.log.error(f"[CAMERA] {self.spec.name}: {self.error}")
            self.stop()
            return False
        self.error = None
        self._gate = FpsGate(self.spec.fps)
        self.log.info(
            f"[CAMERA] {self.spec.name}: {self.spec.prim_path} {self.spec.width}x{self.spec.height} at "
            f"{self.spec.fps:g} fps on port {self.spec.pub_port}, frame_id {self.spec.frame_id}, "
            f"fx=fy={self._intrinsics['fx']:.2f} cx={self._intrinsics['cx']:.1f} cy={self._intrinsics['cy']:.1f} "
            f"(horizontal field of view {horizontal_fov_deg(self._focal_mm, self._aperture_mm):.2f} deg)")
        return True

    def stop(self) -> None:
        for annot in (self._annot_rgb, self._annot_depth):
            if annot is None:
                continue
            try:
                annot.detach()
            except Exception as e:  # noqa: BLE001
                self.log.debug(f"[CAMERA] {self.spec.name}: annotator detach: {e}")
        self._annot_rgb = self._annot_depth = None
        if self._rp is not None:
            try:
                self._rp.destroy()
            except Exception as e:  # noqa: BLE001
                self.log.debug(f"[CAMERA] {self.spec.name}: render product destroy: {e}")
            self._rp = None
        if self._pub is not None:
            self._pub.release()
            self._pub = None
        if self._published or self._dropped or self._failed:
            self.log.info(f"[CAMERA] {self.spec.name}: stopped after {self._published} pairs "
                          f"({self._dropped} dropped, {self._failed} failed)")

    def _read_intrinsics(self, prim) -> None:
        try:
            from pxr import UsdGeom
            cam = UsdGeom.Camera(prim)
            fl = float(cam.GetFocalLengthAttr().Get())
            ha = float(cam.GetHorizontalApertureAttr().Get())
            if not (fl > 0 and ha > 0):
                raise ValueError(f"focalLength {fl}, horizontalAperture {ha}")
            self._focal_mm, self._aperture_mm = fl, ha
        except Exception as e:  # noqa: BLE001
            self.log.warning(f"[CAMERA] {self.spec.name}: intrinsics unreadable ({e}); using Kit's default camera")
        self._intrinsics = intrinsics_from_camera(self._focal_mm, self._aperture_mm, self.spec.width,
                                                  self.spec.height)

    # -- per frame ----------------------------------------------------------
    def on_update(self, now: float) -> None:
        if self._rp is None or not self._gate.due(now):
            return
        rgba = _annotator_array(self._annot_rgb)
        depth = _annotator_array(self._annot_depth)
        if rgba is None or depth is None:
            return                                       # nothing rendered yet
        capture_ns = time.time_ns()
        self.publish(rgba, depth, capture_ns, now)

    def publish(self, rgba: np.ndarray, depth: np.ndarray, capture_unix_ns: int, now: float) -> "Optional[bytes]":
        """Convert one capture, publish it and return the pair blob (None when the buffers do not match)."""
        try:
            rgb = rgba_to_rgb8(rgba)
            mm = depth_to_mm(depth, self._meters_per_unit)
        except Exception as e:  # noqa: BLE001
            self._failed += 1
            if self._failed in (1, 10, 100) or self._failed % 1000 == 0:
                self.log.error(f"[CAMERA] {self.spec.name}: conversion failed ({self._failed}): {e}")
            return None
        if rgb.shape[:2] != mm.shape[:2] or rgb.shape[:2] != (self.spec.height, self.spec.width):
            self._failed += 1
            if not self._size_warned:
                self._size_warned = True
                self.log.warning(f"[CAMERA] {self.spec.name}: rendered rgb {rgb.shape[:2]} / depth {mm.shape[:2]}, "
                                 f"configured {self.spec.height}x{self.spec.width}; frames skipped")
            return None
        self._pair_seq = (self._pair_seq + 1) & 0xFFFFFFFF or 1
        if self._capture_time:
            stamp, flags = int(capture_unix_ns), protocol.FLAG_CAPTURE_UNIX_TIME
        else:
            stamp, flags = 0, 0                          # the camera gateway stamps the arrival time
        blob = protocol.pack_frame_pair(pair_seq=self._pair_seq, depth_mm=mm, rgb=rgb,
                                        frame_id=self.spec.frame_id, observed_at_sync_ns=stamp, flags=flags)
        if self._pub is not None and self._pub.send(self._topic_pair, blob):
            self._published += 1
            self._last_published_at = now
        else:
            self._dropped += 1
        if self._pub is not None and (now - self._last_info_at >= CAMERA_INFO_PERIOD_S or now < self._last_info_at):
            self._last_info_at = now
            self._pub.send(self._topic_info, camera_info_json(self.spec.name, self.spec.frame_id, self.spec.width,
                                                               self.spec.height, self._intrinsics, self.spec.fps))
        if now - self._last_stats_at >= STATS_PERIOD_S:
            self._last_stats_at = now
            valid = float(np.count_nonzero(mm)) / mm.size
            self.log.info(f"[CAMERA] {self.spec.name}: pair {self._pair_seq}, published {self._published}, "
                          f"dropped {self._dropped}, failed {self._failed}, depth valid {100.0 * valid:.0f}%, "
                          f"centre {mm[mm.shape[0] // 2, mm.shape[1] // 2] / 1000.0:.3f} m")
        return blob

    def status(self) -> "Dict[str, Any]":
        return {"name": self.spec.name, "prim_path": self.spec.prim_path, "running": self.running,
                "size": [self.spec.width, self.spec.height], "fps": self.spec.fps, "port": self.spec.pub_port,
                "frame_id": self.spec.frame_id, "published": self._published, "dropped": self._dropped,
                "failed": self._failed, "last_published_at": self._last_published_at, "error": self.error}


def _annotator_array(annot) -> "Optional[np.ndarray]":
    """Annotator readout as a numpy array (a bare array or {'data': array}); None until a frame rendered."""
    if annot is None:
        return None
    try:
        out = annot.get_data()
    except Exception:  # noqa: BLE001 - not rendered yet
        return None
    if isinstance(out, dict):
        out = out.get("data")
    if out is None:
        return None
    arr = np.asarray(out)
    return arr if arr.size else None


# ---------------------------------------------------------------------------
# Every camera of the stage
# ---------------------------------------------------------------------------
class CameraSet:
    """The cameras of the open stage: resolved on Play (the stage may have been edited), rendered and published
    while the timeline plays, released on Stop and when the stage closes."""

    def __init__(self, logger: "Optional[logging.Logger]" = None):
        self.log = logger or _LOG
        self.auto = auto_publish_enabled()
        self.capture_time = capture_time_enabled()
        self.defaults = auto_defaults(logger=self.log)
        self.explicit = load_camera_specs(logger=self.log)
        self.cameras: "List[RcCamera]" = []
        self.skipped: "List[Tuple[str, str]]" = []      # (name, reason) of cameras that did not start
        if self.explicit:
            self.log.info("[CAMERA] explicit entries: " + ", ".join(str(e["name"]) for e in self.explicit))
        if not self.auto:
            self.log.info(f"[CAMERA] automatic publishing off ({config.CAMERA_AUTO_ENV}=0)")
        if not self.capture_time:
            self.log.info(f"[CAMERA] frames carry no capture time ({config.CAMERA_CAPTURE_TIME_ENV}=0): "
                          f"the camera gateways stamp their arrival")

    def start(self, stage) -> None:
        self.stop()
        if stage is None:
            return
        try:
            discovered = discover_cameras(stage, self.log) if self.auto else []
        except Exception as e:  # noqa: BLE001
            self.log.error(f"[CAMERA] camera discovery failed: {e}")
            discovered = []
        specs = resolve_specs(discovered, self.explicit, auto=self.auto, defaults=self.defaults, logger=self.log)
        self.skipped = []
        for spec in specs:
            cam = RcCamera(spec, self.log, capture_time=self.capture_time)
            if cam.start(stage):
                self.cameras.append(cam)
            else:
                self.skipped.append((spec.name, cam.error or "not started"))
        if self.cameras:
            self.log.info("[CAMERA] publishing " + ", ".join(c.spec.name for c in self.cameras))

    def stop(self) -> None:
        for cam in self.cameras:
            try:
                cam.stop()
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"[CAMERA] {cam.spec.name}: stop error: {e}")
        self.cameras = []

    def on_update(self) -> None:
        if not self.cameras:
            return
        now = time.monotonic()
        for cam in self.cameras:
            try:
                cam.on_update(now)
            except Exception as e:  # noqa: BLE001 - a render hiccup must never stop the update stream
                cam._failed += 1
                if cam._failed in (1, 10, 100):
                    self.log.error(f"[CAMERA] {cam.spec.name}: capture error: {e}")

    def status(self) -> "List[Dict[str, Any]]":
        return [c.status() for c in self.cameras] + [
            {"name": name, "running": False, "error": reason} for name, reason in self.skipped]
