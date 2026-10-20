"""Wire-byte compatibility across the packers' allocation-free copy path.

The struct encoders remain an independent oracle. Reuse each packer across
changing input lengths to catch stale state in both data and padding bytes.
"""
import warnings

import numpy as np
import pytest

from scripts import protocol

# numpy 2 moved ComplexWarning into numpy.exceptions
_COMPLEX_WARNING = getattr(getattr(np, "exceptions", np), "ComplexWarning")

DTYPES = ["f2", "f4", "f8", ">f4", ">f8", "i8", "u8", "?", "O", "U24", "c16"]


def _values(n, dtype):
    values = np.arange(n, dtype=np.float64) / 4
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_state_bytes_for_dtype_and_changing_lengths(dtype):
    packer = protocol.StatePacker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _COMPLEX_WARNING)
        for seq, n in enumerate((30, 24, 18, 12, 1, 0, 24, 3, 18, 0)):
            vectors = [_values(k, dtype) for k in (n, n + 2, max(0, n - 1))]
            stamp = -123456789 if seq % 2 else 9876543210
            wire = packer.pack(stamp, seq, n, *vectors)
            assert wire == protocol.pack_state(stamp, seq, n, *vectors)
            assert len(wire) == protocol.SIM_STATE_SIZE
            assert wire[20:24] == bytes(4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_imu_bytes_for_dtype_and_changing_lengths(dtype):
    packer = protocol.ImuPacker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _COMPLEX_WARNING)
        for seq, n in enumerate((7, 4, 3, 1, 0, 5, 0)):
            vectors = [_values(k, dtype) for k in (n, n + 1, n + 2, n, n)]
            for contacts in ([], [False], [0, -3], [True, False, True], [np.nan, 0]):
                wire = packer.pack(-1234, seq, *vectors, contacts)
                assert wire == protocol.pack_imu_state(-1234, seq, *vectors, contacts)
                assert len(wire) == protocol.IMU_STATE_SIZE
                assert wire[-6:] == bytes(6)


@pytest.mark.parametrize("kind", ["slice", "negative_stride", "readonly", "matrix", "list", "scalar", "subclass"])
def test_array_layout_and_sequence_fallbacks(kind):
    values = np.arange(36, dtype=np.float32)
    if kind == "slice":
        values = values[::2]
    elif kind == "negative_stride":
        values = values[::-1]
    elif kind == "readonly":
        values.setflags(write=False)
    elif kind == "matrix":
        values = values.reshape(6, 6)
    elif kind == "list":
        values = values.reshape(6, 6).tolist()
    elif kind == "scalar":
        values = 1.25
    elif kind == "subclass":
        values = np.ma.array(values, mask=True)
    state = protocol.StatePacker()
    assert state.pack(1, 2, 18, values, values, values) == protocol.pack_state(1, 2, 18, values, values, values)
    imu = protocol.ImuPacker()
    assert imu.pack(1, 2, *([values] * 5), [1, 0]) == protocol.pack_imu_state(1, 2, *([values] * 5), [1, 0])


@pytest.mark.parametrize("dtype,bits", [
    ("f4", [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000,
             0xff800000, 0x7fc00123, 0xffc00123, 0x7f800123, 0xff800123]),
    ("f8", [0x0000000000000000, 0x8000000000000000, 0x0000000000000001,
             0x8000000000000001, 0x7ff0000000000000, 0xfff0000000000000,
             0x7ff8000000000123, 0xfff8000000000123, 0x7ff0000000000123,
             0xfff0000000000123]),
])
def test_float_bit_patterns_match_previous_numpy_conversion(dtype, bits):
    values = np.asarray(bits, dtype="u" + dtype[1:]).view(dtype)
    state, imu = protocol.StatePacker(), protocol.ImuPacker()
    with np.errstate(invalid="ignore"):
        # Compare with the original NumPy cast, retaining float64 NaN payloads
        # directly instead of routing them through Python floats in struct.pack.
        widened = np.asarray(values, dtype=np.float64)
        expected = np.zeros(protocol.SIM_MAX_JOINTS, dtype="<f8")
        expected[:len(values)] = widened
        wire = state.pack(17, 23, len(values), values, values, values)
        assert wire[24:] == expected.tobytes() * 3
        wire = imu.pack(17, 23, values, values, values, values, values, [1, 0])
        assert wire[16:144] == widened[:4].astype("<f8").tobytes() + widened[:3].astype("<f8").tobytes() * 4


def test_wire_headers_preserve_full_integer_ranges():
    state, imu = protocol.StatePacker(), protocol.ImuPacker()
    for stamp in (-(2**63), 0, 2**63 - 1):
        for seq in (0, 2**63, 2**64 - 1):
            assert state.pack(stamp, seq, -1, [], [], []) == protocol.pack_state(stamp, seq, -1, [], [], [])
            assert imu.pack(stamp, seq, [], [], [], [], [], []) == protocol.pack_imu_state(stamp, seq, [], [], [], [], [], [])


def test_returned_bytes_remain_immutable_after_next_pack():
    state, imu = protocol.StatePacker(), protocol.ImuPacker()
    vectors = [np.arange(24, dtype=np.float32)] * 5
    first_state = state.pack(1, 2, 24, *vectors[:3])
    first_imu = imu.pack(1, 2, *vectors, [1, 1])
    state.pack(3, 4, 0, [], [], [])
    imu.pack(3, 4, [], [], [], [], [], [])
    assert first_state == protocol.pack_state(1, 2, 24, *vectors[:3])
    assert first_imu == protocol.pack_imu_state(1, 2, *vectors, [1, 1])
