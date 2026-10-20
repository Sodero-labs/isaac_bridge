"""World contact-force pass: one RigidContactView read, per-robot contact_forces payloads in link order."""
from __future__ import annotations

import numpy as np

from scripts.contact_forces import ContactForceBatch
from scripts import protocol


class _Device:
    def __init__(self):
        self.payload = None

    def set_contact_force_payload(self, blob):
        self.payload = blob


class _ContactView:
    def __init__(self, paths, forces):
        self.sensor_paths = list(reversed(paths))     # the view may order sensors differently
        self._forces = forces
        self.dt_seen = None

    def get_net_contact_forces(self, dt):
        self.dt_seen = dt
        return np.array([self._forces[p] for p in self.sensor_paths], dtype=np.float32)


class _SimView:
    def __init__(self, forces):
        self.forces = forces
        self.is_valid = True
        self.created = 0
        self.view = None

    def create_rigid_contact_view(self, paths):
        self.created += 1
        self.view = _ContactView(paths, self.forces)
        return self.view


def _paths(robot, links):
    return [f"/World/{robot}/links/{name}" for name in links]


def test_payloads_follow_each_robots_link_order_and_z_component():
    a, b = _paths("ur3_1", ["ll_faa", "lr_faa", "ur3_wrist3"]), _paths("ur3_2", ["ll_faa", "lr_faa"])
    forces = {p: (0.1 * i, -0.2 * i, 10.0 * (i + 1)) for i, p in enumerate(a + b)}
    sim = _SimView(forces)
    da, db = _Device(), _Device()
    batch = ContactForceBatch([(da, a), (db, b)], sim_view_factory=lambda: sim)
    assert batch.compute(1.25, 0.004)
    stamp, seq, fa = protocol.parse_contact_forces(da.payload)
    assert (stamp, seq) == (1_250_000_000, 1)
    np.testing.assert_allclose(fa, [10.0, 20.0, 30.0])
    _, _, fb = protocol.parse_contact_forces(db.payload)
    np.testing.assert_allclose(fb, [40.0, 50.0])
    assert sim.view.dt_seen == 0.004


def test_rebinds_only_when_the_simulation_view_changes():
    paths = _paths("ur3_1", ["ll_faa"])
    sims = [_SimView({paths[0]: (0, 0, 5.0)})]
    device = _Device()
    batch = ContactForceBatch([(device, paths)], sim_view_factory=lambda: sims[-1])
    assert batch.compute(0.0, 0.004) and batch.compute(0.004, 0.004)
    assert sims[-1].created == 1
    sims.append(_SimView({paths[0]: (0, 0, 7.0)}))          # timeline play: new SimulationView
    assert batch.compute(0.008, 0.004)
    assert sims[-1].created == 1
    assert protocol.parse_contact_forces(device.payload)[2][0] == 7.0


def test_no_physics_view_publishes_nothing():
    device = _Device()
    batch = ContactForceBatch([(device, _paths("ur3_1", ["ll_faa"]))], sim_view_factory=lambda: None)
    assert not batch.compute(0.0, 0.004)
    assert device.payload is None


def test_wire_record_round_trip_and_size():
    blob = protocol.pack_contact_forces(42, 3, [1.0, 2.5])
    assert len(blob) == protocol.CONTACT_FORCES_SIZE == 56
    stamp, seq, forces = protocol.parse_contact_forces(blob)
    assert (stamp, seq) == (42, 3)
    np.testing.assert_allclose(forces, [1.0, 2.5])
