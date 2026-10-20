"""Camera frames and the camera module — no Isaac Sim required.

Pins the Python side of the camera wire format to the camera gateway's C++ contract (rc_camera/frame_protocol.hpp):

  1. FrameHdr = "<4s7IqfI16s", 64 B, field offsets == the C++ struct layout.
  2. Header fields round-trip; frame_id is NUL-terminated and length-checked.
  3. Pair blob = [hdr depth][depth16][hdr rgb][rgb8] splits and parses back.
  4. depth m -> mm conversion rounds, clamps and maps invalid to 0.
  5. The camera module's Isaac-free half: naming and configuration (the cloud renderer's rules), conversion, the
     capture-time flag, and the ZMQ envelope ``TM/<name>/camera_pair||blob`` as the camera gateway's SUB sees it.
"""
import json
import os
import re
import struct
import time

import numpy as np
import pytest

from scripts import protocol
from scripts.device import camera as cam_mod      # must import without Isaac Sim
from scripts.device.camera import CameraSpec, RcCamera, load_camera_specs, resolve_specs

# The camera gateway's header, when its source is checked out next to this folder (else that test is skipped)
CPP_HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "camera_gateway", "include", "rc_camera", "frame_protocol.hpp")


# ---------------------------------------------------------------------------
# 1. layout
# ---------------------------------------------------------------------------
def test_header_is_64_bytes_and_matches_cpp_layout():
    assert protocol.FRAME_HDR_FMT == "<4s7IqfI16s"
    assert protocol.FRAME_HDR_SIZE == 64
    assert struct.calcsize(protocol.FRAME_HDR_FMT) == 64

    # byte offsets of each field, as the C++ compiler lays them out (natural alignment,
    # no padding needed: 4+4*7=32 -> i64 at 32 -> f32 at 40 -> u32 at 44 -> char[16] at 48)
    expected = {
        "magic": 0, "version": 4, "pair_seq": 8, "width": 12, "height": 16, "step_bytes": 20,
        "payload_bytes": 24, "encoding": 28, "observed_at_sync_ns": 32, "depth_unit_m": 40,
        "flags": 44, "frame_id": 48,
    }
    hdr = protocol.pack_frame_hdr(
        pair_seq=0x01020304, width=0x11121314, height=0x21222324, encoding=protocol.ENCODING_DEPTH16,
        payload_bytes=0x41424344, step_bytes=0x31323334, frame_id="cam0_optical",
        observed_at_sync_ns=0x5152535455565758, depth_unit_m=0.001, flags=0x61626364)
    assert hdr[0:4] == b"RCIM"
    assert struct.unpack_from("<I", hdr, expected["version"])[0] == 1
    assert struct.unpack_from("<I", hdr, expected["pair_seq"])[0] == 0x01020304
    assert struct.unpack_from("<I", hdr, expected["width"])[0] == 0x11121314
    assert struct.unpack_from("<I", hdr, expected["height"])[0] == 0x21222324
    assert struct.unpack_from("<I", hdr, expected["step_bytes"])[0] == 0x31323334
    assert struct.unpack_from("<I", hdr, expected["payload_bytes"])[0] == 0x41424344
    assert struct.unpack_from("<I", hdr, expected["encoding"])[0] == 2
    assert struct.unpack_from("<q", hdr, expected["observed_at_sync_ns"])[0] == 0x5152535455565758
    assert struct.unpack_from("<f", hdr, expected["depth_unit_m"])[0] == pytest.approx(0.001)
    assert struct.unpack_from("<I", hdr, expected["flags"])[0] == 0x61626364
    assert hdr[expected["frame_id"]:] == b"cam0_optical\0\0\0\0"


def test_constants_match_cpp_header_when_present():
    """Cross-check enum values and sizes against the C++ header text."""
    if not os.path.exists(CPP_HEADER):
        pytest.skip("frame_protocol.hpp not found")
    src = open(CPP_HEADER, encoding="utf-8").read()
    assert 'FrameHdr = "<4s7IqfI16s"' in src
    assert "sizeof(FrameHdr) == 64" in src
    m = re.search(r"RGB8\s*=\s*(\d+)", src); assert int(m.group(1)) == protocol.ENCODING_RGB8
    m = re.search(r"DEPTH16\s*=\s*(\d+)", src); assert int(m.group(1)) == protocol.ENCODING_DEPTH16
    m = re.search(r"DEPTH32F\s*=\s*(\d+)", src); assert int(m.group(1)) == protocol.ENCODING_DEPTH32F
    m = re.search(r"kNotifyOnly\s*=\s*1u\s*<<\s*(\d+)", src); assert 1 << int(m.group(1)) == protocol.FLAG_NOTIFY_ONLY
    m = re.search(r"kDepthFirst\s*=\s*1u\s*<<\s*(\d+)", src); assert 1 << int(m.group(1)) == protocol.FLAG_DEPTH_FIRST
    m = re.search(r"kProtocolVersion\s*=\s*(\d+)", src); assert int(m.group(1)) == protocol.FRAME_PROTOCOL_VERSION
    assert "char     frame_id[16]" in src


# ---------------------------------------------------------------------------
# 2. header round-trip
# ---------------------------------------------------------------------------
def test_header_fields_round_trip():
    hdr = protocol.pack_frame_hdr(pair_seq=77, width=640, height=480, encoding=protocol.ENCODING_RGB8,
                                  payload_bytes=640 * 480 * 3, frame_id="cam_arm_opt",
                                  observed_at_sync_ns=-5, flags=protocol.FLAG_DEPTH_FIRST)
    h = protocol.unpack_frame_hdr(hdr)
    assert h.is_valid
    assert (h.pair_seq, h.width, h.height) == (77, 640, 480)
    assert h.step_bytes == 640 * 3            # defaulted from encoding
    assert h.payload_bytes == 640 * 480 * 3
    assert h.encoding == protocol.ENCODING_RGB8
    assert h.depth_unit_m == 0.0              # rgb: no depth unit
    assert h.observed_at_sync_ns == -5
    assert h.flags == protocol.FLAG_DEPTH_FIRST
    assert h.frame_id == "cam_arm_opt"

    d = protocol.unpack_frame_hdr(protocol.pack_frame_hdr(
        pair_seq=1, width=320, height=240, encoding=protocol.ENCODING_DEPTH16,
        payload_bytes=320 * 240 * 2, frame_id="c"))
    assert d.step_bytes == 320 * 2 and d.depth_unit_m == pytest.approx(0.001)

    # pair_seq wraps to u32 like the C++ field
    w = protocol.unpack_frame_hdr(protocol.pack_frame_hdr(
        pair_seq=2 ** 32 + 5, width=1, height=1, encoding=protocol.ENCODING_RGB8,
        payload_bytes=3, frame_id="x"))
    assert w.pair_seq == 5


def test_header_frame_id_limits_and_bad_input():
    ok = protocol.pack_frame_hdr(pair_seq=1, width=1, height=1, encoding=protocol.ENCODING_RGB8,
                                 payload_bytes=3, frame_id="a" * 15)
    assert protocol.unpack_frame_hdr(ok).frame_id == "a" * 15
    with pytest.raises(ValueError):   # 16 chars would leave no NUL for the C++ reader
        protocol.pack_frame_hdr(pair_seq=1, width=1, height=1, encoding=protocol.ENCODING_RGB8,
                                payload_bytes=3, frame_id="a" * 16)
    with pytest.raises(ValueError):   # unknown encoding needs an explicit step
        protocol.pack_frame_hdr(pair_seq=1, width=1, height=1, encoding=9, payload_bytes=0, frame_id="x")
    with pytest.raises(ValueError):
        protocol.unpack_frame_hdr(b"RCIM" + b"\0" * 10)       # too short
    with pytest.raises(ValueError):
        protocol.unpack_frame_hdr(b"XXXX" + b"\0" * 60)       # bad magic


# ---------------------------------------------------------------------------
# 3. pair blob
# ---------------------------------------------------------------------------
def _synthetic_pair(h=12, w=16):
    depth_mm = (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 7) % 5000
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(w, dtype=np.uint8)
    rgb[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    rgb[..., 2] = 200
    return depth_mm, rgb


def test_pair_blob_split_and_parse():
    depth_mm, rgb = _synthetic_pair()
    h, w = depth_mm.shape
    blob = protocol.pack_frame_pair(pair_seq=42, depth_mm=depth_mm, rgb=rgb, frame_id="cam0_optical",
                                    observed_at_sync_ns=0)
    assert len(blob) == 2 * 64 + h * w * 2 + h * w * 3

    frames = protocol.split_frames(blob)
    assert len(frames) == 2
    (dh, dpx), (rh, rpx) = frames
    # depth first, both share pair_seq / frame_id / flags
    assert dh.encoding == protocol.ENCODING_DEPTH16 and rh.encoding == protocol.ENCODING_RGB8
    assert dh.pair_seq == rh.pair_seq == 42
    assert dh.frame_id == rh.frame_id == "cam0_optical"
    assert dh.flags & protocol.FLAG_DEPTH_FIRST and rh.flags & protocol.FLAG_DEPTH_FIRST
    assert dh.observed_at_sync_ns == 0 and rh.observed_at_sync_ns == 0   # gateway fills it in
    assert (dh.width, dh.height, dh.step_bytes, dh.payload_bytes) == (w, h, w * 2, h * w * 2)
    assert (rh.width, rh.height, rh.step_bytes, rh.payload_bytes) == (w, h, w * 3, h * w * 3)
    assert dh.depth_unit_m == pytest.approx(0.001) and rh.depth_unit_m == 0.0

    np.testing.assert_array_equal(protocol.frame_to_array(dh, dpx), depth_mm)
    np.testing.assert_array_equal(protocol.frame_to_array(rh, rpx), rgb)

    # C++ reader view: second header sits exactly after the first frame's pixels
    off2 = 64 + h * w * 2
    assert blob[off2:off2 + 4] == b"RCIM"

    # truncated blob is rejected rather than silently yielding a short frame
    with pytest.raises(ValueError):
        protocol.split_frames(blob[:-1])

    # non-contiguous input (a view) packs identically
    big = np.zeros((h, 2 * w, 3), dtype=np.uint8); big[:, :w] = rgb
    blob2 = protocol.pack_frame_pair(pair_seq=42, depth_mm=depth_mm, rgb=big[:, :w], frame_id="cam0_optical")
    assert blob2 == blob

    with pytest.raises(ValueError):   # size mismatch between depth and rgb
        protocol.pack_frame_pair(pair_seq=1, depth_mm=depth_mm[:-1], rgb=rgb, frame_id="x")
    with pytest.raises(ValueError):   # wrong dtype for DEPTH16
        protocol.pack_frame(pair_seq=1, encoding=protocol.ENCODING_DEPTH16,
                            pixels=depth_mm.astype(np.float32), frame_id="x")


# ---------------------------------------------------------------------------
# 4. depth conversion
# ---------------------------------------------------------------------------
def test_depth_m_to_mm_rounds_and_clamps():
    d = np.array([[0.0, 1.2344, 1.2346, 0.0004, 0.0006],
                  [np.nan, np.inf, -np.inf, -0.5, 70.0],
                  [65.535, 65.5354, 65.536, 6.0, 0.1]], dtype=np.float32)
    mm = protocol.depth_m_to_mm(d)
    assert mm.dtype == np.uint16 and mm.shape == d.shape
    assert mm[0].tolist() == [0, 1234, 1235, 0, 1]           # round-to-nearest mm
    assert mm[1].tolist() == [0, 0, 0, 0, 65535]              # invalid -> 0, far finite -> saturate
    assert mm[2].tolist() == [65535, 65535, 65535, 6000, 100]
    # float64 input also works; output is a fresh array (input untouched)
    d64 = np.full((2, 2), 2.5, dtype=np.float64)
    assert protocol.depth_m_to_mm(d64).tolist() == [[2500, 2500], [2500, 2500]]
    assert d64[0, 0] == 2.5


# ---------------------------------------------------------------------------
# 5. camera module (Isaac-free part) + ZMQ envelope
# ---------------------------------------------------------------------------
def test_camera_module_imports_without_isaac():
    assert cam_mod.RGB_ANNOTATOR == "rgb" and cam_mod.DEPTH_ANNOTATOR == "distance_to_image_plane"
    cam = RcCamera(CameraSpec.from_dict({"name": "cam0", "prim_path": "/World/r/cam0"}))
    assert not cam.running and cam.status()["published"] == 0


def test_names_and_frame_ids_follow_the_cloud_renderer():
    assert cam_mod.topic_safe_name("cam0") == "cam0"
    assert cam_mod.topic_safe_name("front cam/1") == "front_cam_1"
    assert cam_mod.topic_safe_name("") == "camera"
    assert cam_mod.default_frame_id("cam0") == "cam0_optical"
    long_id = cam_mod.default_frame_id("a_very_long_camera_name")
    assert long_id.endswith("_optical") and len(long_id.encode()) == protocol.FRAME_ID_MAX_LEN
    d = CameraSpec.from_dict({"name": "n", "prim_path": "/p"})
    assert (d.width, d.height, d.fps, d.frame_id, d.pub_port) == (640, 480, 15.0, "n_optical", 53335)
    for bad in ({"name": "a/b", "prim_path": "/p"}, {"name": "a", "prim_path": "p"},
                {"name": "a", "prim_path": "/p", "fps": 0}, {"name": "a", "prim_path": "/p", "width": 0},
                {"name": "a", "prim_path": "/p", "frame_id": "x" * 16}, {"prim_path": "/p"}):
        with pytest.raises(ValueError):
            CameraSpec.from_dict(bad)


def test_explicit_entries_from_config_and_env():
    assert load_camera_specs(config_list=[], env_value="") == []
    cfg = [{"name": "a", "prim_path": "/World/a/Camera"},
           {"name": "bad", "prim_path": "/x", "fps": 0},               # skipped
           {"name": "b/c", "prim_path": "/x"}]                         # '/' in the topic: skipped
    env = json.dumps([{"name": "a", "prim_path": "/World/a/Cam2", "width": 320, "height": 240,
                       "fps": 10, "frame_id": "cam0_optical", "pub_port": 53400},
                      {"name": "front", "width": 424}])                 # override only: no prim_path needed
    entries = load_camera_specs(config_list=cfg, env_value=env)
    assert [e["name"] for e in entries] == ["a", "front"]
    assert entries[0]["prim_path"] == "/World/a/Cam2" and entries[0]["pub_port"] == 53400
    # malformed env is ignored, the config list still applies
    assert len(load_camera_specs(config_list=[{"name": "n", "prim_path": "/p"}], env_value="{not json")) == 1


def test_resolve_specs_auto_cameras_and_overrides():
    discovered = [("cam0", "/World/tb3/base_link/cam0"), ("front", "/World/r/front")]
    defaults = {"width": 640, "height": 480, "fps": 15.0, "pub_port": 53335}
    explicit = [{"name": "front", "width": 424, "height": 240},                 # override the automatic one
                {"name": "extra", "prim_path": "/World/x/cam", "fps": 5},        # an extra camera
                {"name": "ghost", "width": 320}]                                 # nothing to override: skipped
    specs = {s.name: s for s in resolve_specs(discovered, explicit, auto=True, defaults=defaults)}
    assert set(specs) == {"cam0", "front", "extra"}
    assert specs["cam0"] == CameraSpec("cam0", "/World/tb3/base_link/cam0", 640, 480, 15.0, "cam0_optical", 53335)
    assert (specs["front"].width, specs["front"].height, specs["front"].prim_path) == (424, 240, "/World/r/front")
    assert specs["extra"].fps == 5.0 and specs["extra"].frame_id == "extra_optical"
    only_explicit = resolve_specs(discovered, explicit, auto=False, defaults=defaults)
    assert [s.name for s in only_explicit] == ["extra"]


class _Prim:
    def __init__(self, path, camera=True):
        self._path, self._camera = path, camera

    def GetPath(self):
        return self._path

    def GetName(self):
        return self._path.rsplit("/", 1)[-1]

    def IsA(self, schema):
        from pxr import UsdGeom
        return self._camera and schema is UsdGeom.Camera


class _Stage:
    def __init__(self, prims):
        self._prims = prims

    def Traverse(self):
        return iter(self._prims)


def test_discovery_skips_kit_cameras_and_ambiguous_names():
    stage = _Stage([_Prim("/OmniverseKit_Persp"), _Prim("/Render/OmniverseKit/HydraTextures/cam"),
                    _Prim("/World/tb3/links/base_link/cam0"), _Prim("/World/light", camera=False),
                    _Prim("/World/a/eye"), _Prim("/World/b/eye")])
    assert cam_mod.discover_cameras(stage) == [("cam0", "/World/tb3/links/base_link/cam0")]


def test_auto_defaults_from_env():
    env = {"RC_CAMERA_AUTO_WIDTH": "320", "RC_CAMERA_AUTO_HEIGHT": "240", "RC_CAMERA_AUTO_FPS": "abc"}
    d = cam_mod.auto_defaults(env)
    assert (d["width"], d["height"], d["fps"]) == (320, 240, 15.0)     # a bad value falls back, never raises
    assert cam_mod.auto_publish_enabled({}) is True
    assert cam_mod.auto_publish_enabled({"RC_CAMERA_AUTO_PUBLISH": "0"}) is False


def test_fps_gate_on_the_wall_clock():
    gate = cam_mod.FpsGate(10.0)                                   # 100 ms
    hits = [t for t in np.arange(0.0, 1.0, 1 / 60.0) if gate.due(float(t))]
    assert len(hits) == 10 and np.allclose(np.diff(hits), 0.1, atol=1 / 60.0 + 1e-9)
    assert gate.due(1.55) and not gate.due(1.56)                   # a 0.5 s stall: one capture, no burst


def test_intrinsics_and_field_of_view():
    intr = cam_mod.intrinsics_from_camera(18.147562, 20.955, 640, 480)   # Kit's default camera
    assert intr["fx"] == pytest.approx(554.256, rel=1e-5) and intr["fy"] == intr["fx"]
    assert (intr["cx"], intr["cy"]) == (320.0, 240.0)
    assert cam_mod.horizontal_fov_deg(18.147562, 20.955) == pytest.approx(60.0, abs=1e-4)


def test_conversions():
    rgba = np.zeros((2, 3, 4), np.uint8); rgba[..., 0] = 7; rgba[..., 3] = 255
    rgb = cam_mod.rgba_to_rgb8(rgba)
    assert rgb.shape == (2, 3, 3) and rgb.flags["C_CONTIGUOUS"] and rgb[0, 0].tolist() == [7, 0, 0]
    depth = np.array([[1.0, np.inf], [np.nan, 0.5]], np.float32)[..., None]   # (H, W, 1) is accepted
    assert cam_mod.depth_to_mm(depth, 1.0).tolist() == [[1000, 0], [0, 500]]
    assert cam_mod.depth_to_mm(np.full((1, 1), 150.0, np.float32), 0.01).tolist() == [[1500]]   # centimetre stage
    with pytest.raises(ValueError):
        cam_mod.rgba_to_rgb8(np.zeros((2, 2), np.uint8))


def test_publish_packs_capture_time_and_reaches_a_gateway_socket():
    import zmq
    h, w = 12, 16
    port = 63335
    spec = CameraSpec.from_dict({"name": "cam0", "prim_path": "/World/tb3/cam0", "width": w, "height": h,
                                 "pub_port": port})
    cam = RcCamera(spec)
    cam._pub = cam_mod._PairPublisher.acquire(port)            # the Isaac-free half of start()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)                                  # == camera_gateway isaac input
    try:
        sub.set(zmq.CONFLATE, 1)
        sub.set(zmq.RCVTIMEO, 3000)
        sub.setsockopt(zmq.SUBSCRIBE, b"TM/cam0/camera_pair")
        sub.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.3)                                        # PUB/SUB slow joiner

        depth = np.full((h, w), 1.5, dtype=np.float32)
        depth[0, 0] = np.inf                                   # no hit
        depth[1, 1] = 0.2506
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 10; rgba[..., 1] = 20; rgba[..., 2] = 30; rgba[..., 3] = 255
        stamp = time.time_ns()
        raw = None
        deadline = time.time() + 3.0
        while time.time() < deadline and raw is None:
            assert cam.publish(rgba, depth, stamp, time.monotonic()) is not None
            try:
                raw = sub.recv()
            except zmq.Again:
                continue
        assert raw is not None, "the gateway-side SUB received nothing"
        topic, payload = protocol.split_zmq_frame(raw)
        assert topic == b"TM/cam0/camera_pair"
        (dh, dpx), (rh, rpx) = protocol.split_frames(payload)
        assert dh.pair_seq == rh.pair_seq >= 1 and (dh.width, dh.height) == (w, h)
        assert dh.flags == rh.flags == protocol.FLAG_DEPTH_FIRST | protocol.FLAG_CAPTURE_UNIX_TIME
        assert dh.observed_at_sync_ns == rh.observed_at_sync_ns == stamp
        assert dh.frame_id == rh.frame_id == "cam0_optical"
        d = protocol.frame_to_array(dh, dpx)
        assert d[0, 0] == 0 and d[1, 1] == 251 and d[5, 5] == 1500
        assert protocol.frame_to_array(rh, rpx)[3, 3].tolist() == [10, 20, 30]          # alpha stripped

        # a frame of the wrong size is skipped, the sequence untouched
        seq = dh.pair_seq
        assert cam.publish(np.zeros((h + 1, w, 4), np.uint8), np.zeros((h + 1, w), np.float32), stamp,
                           time.monotonic()) is None
        assert cam._pair_seq == seq

        # camera_info: the cloud renderer's shape, small
        info = json.loads(cam_mod.camera_info_json("cam0", "cam0_optical", w, h,
                                                   cam_mod.intrinsics_from_camera(24.0, 20.955, w, h), 15.0))
        assert set(info) == {"name", "frame_id", "width", "height", "fps", "intrinsics", "distortion_model"}
        assert info["intrinsics"]["fx"] == pytest.approx(24.0 * w / 20.955, rel=1e-4)
    finally:
        sub.close(linger=0); ctx.term()
        cam.stop()


def test_frames_without_capture_time_on_request():
    assert cam_mod.capture_time_enabled({}) is True
    assert cam_mod.capture_time_enabled({"RC_CAMERA_CAPTURE_TIME": "0"}) is False
    spec = CameraSpec.from_dict({"name": "cam0", "prim_path": "/p", "width": 4, "height": 2})
    cam = RcCamera(spec, capture_time=False)                     # no socket: publish still packs and counts
    blob = cam.publish(np.zeros((2, 4, 4), np.uint8), np.ones((2, 4), np.float32), time.time_ns(), 0.0)
    (dh, _), (rh, _) = protocol.split_frames(blob)
    assert dh.flags == rh.flags == protocol.FLAG_DEPTH_FIRST     # the gateway stamps the arrival time
    assert dh.observed_at_sync_ns == rh.observed_at_sync_ns == 0
