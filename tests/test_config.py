"""Settings read from the environment."""
import logging

import pytest

from scripts import config
from scripts.sim_rates import snap_render_hz, world_rates


@pytest.mark.parametrize("raw, default, expected", [
    (None, True, True), ("", False, False), ("1", False, True), ("yes", False, True), ("ON", False, True),
    ("0", True, False), ("off", True, False), ("false", True, False), ("anything", True, False),
])
def test_env_flag(raw, default, expected):
    env = {} if raw is None else {"X": raw}
    assert config.env_flag("X", default, env) is expected


def test_robot_settings_defaults():
    assert config.respawn_at_bind({}) is True
    assert config.frame_latch_enabled({}) is False
    assert config.respawn_at_bind({config.RESPAWN_AT_BIND_ENV: "0"}) is False
    assert config.frame_latch_enabled({config.FRAME_LATCH_ENV: "1"}) is True


def test_contact_height_override(caplog):
    assert config.contact_height_override({}) is None
    assert config.contact_height_override({config.CONTACT_HEIGHT_ENV: "0.005"}) == 0.005
    with caplog.at_level(logging.ERROR, logger="isaac_bridge"):
        assert config.contact_height_override({config.CONTACT_HEIGHT_ENV: "5mm"}) is None
        assert config.contact_height_override({config.CONTACT_HEIGHT_ENV: "-1"}) is None
    assert caplog.text.count("ignored") == 2


def test_world_rates_follow_env_then_world_then_default():
    assert world_rates(None, {}) == (60.0, 60.0)
    assert world_rates(None, {config.PHYSICS_HZ_ENV: "200"}) == (200.0, 50.0)       # 4 physics steps per frame
    assert world_rates(None, {config.PHYSICS_HZ_ENV: "250", config.RENDER_HZ_ENV: "60"}) == (250.0, 50.0)
    assert snap_render_hz(100.0, 30.0) == 25.0
    assert snap_render_hz(60.0, 120.0) == 60.0                                       # never above physics
