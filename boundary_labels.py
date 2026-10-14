"""Map and statistics labels taken from the mechanism that actually runs.

The kinematic code in `s.bcode` is a gate for processes: a boundary is called
divergent, convergent or sliding by comparing its normal speed with
`max(2 km/Myr, 0.15 * sliding speed)`. Normal partition version 1 then subducts
and collides on any closing motion, whatever that gate said. The map and the
length statistics read the gate, so a margin the engine is subducting could be
drawn as a transform, and inherited trenches needed a separate override to look
like subduction at all.

These labels answer a different question: what is this boundary doing? A closing
edge with a downgoing plate and water coming in is subduction. A closing edge
with continental material on both sides, or one whose incoming plate has become
buoyant and jammed the margin, is collision. Spreading keeps the exact
classification of the production path. Everything else that still exists is
sliding. `s.bcode` is untouched, so no process changes; only what is displayed,
counted and exported does. See IMPLEMENTATION_GAPS.md G116 and G117.

Version 2 adds one label. "Everything else" above includes ocean floor that is
closing where no trench runs: no downgoing plate has been declared, so nothing
there is subducting, and no buoyant crust is involved, so it is not a collision.
Version 1 draws it as a transform, which reads as a fault sliding sideways while
the plates on either side approach by kilometres every million years. Version 2
calls it converging without a trench. The test is the kinematic gate itself: such a
boundary is relabelled only where `s.bcode` says the motion is convergent, so a
fault that slides with a little closing across it is still a transform.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np

import normal_partition

VERSION = 1
CONVERGING_VERSION = 2
NONE, RIDGE, SUBDUCTION, TRANSFORM, COLLISION, RIFT, CONVERGING = range(7)
MINIMUM_LENGTH_KM = 1e-10
# A control cell takes the label of highest rank among its contacts: collision over rift over
# subduction over converging without a trench over ridge over transform. The second table reads
# a rank back as its label.
RANK = np.array([0, 2, 4, 1, 6, 5, 3], np.uint8)
RANKED = np.array([0, 3, 1, 6, 2, 5, 4], np.uint8)


def version(s):
    value = getattr(s, 'boundary_label_version', 0)
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value not in (0, VERSION, CONVERGING_VERSION)):
        raise ValueError('Unsupported boundary label version.')
    return int(value)


def enabled(s):
    return version(s) > 0


def upgrade(s, target=VERSION):
    """Select mechanism labels. Display only: no physical state changes."""
    if target not in (VERSION, CONVERGING_VERSION) or isinstance(target, (bool, np.bool_)):
        raise ValueError('Unsupported boundary label version.')
    current = version(s)
    if current >= target:
        return deepcopy(getattr(s, 'boundary_label_migration', {}))
    report = dict(version=VERSION, time_myr=float(s.t),
                  selection='Map, statistics and exports label each boundary by the mechanism that runs on it.',
                  physical_state_changed=False, kinematic_code_retained=True)
    if target == CONVERGING_VERSION:
        report = dict(report, version=CONVERGING_VERSION,
                      selection=report['selection']+' Closing boundaries with no trench and no collision are'
                                ' labelled converging without a trench rather than transform.')
        if current:
            report['predecessor'] = deepcopy(getattr(s, 'boundary_label_migration', {}))
    s.boundary_label_version = target
    s.boundary_label_migration = report
    return deepcopy(report)


def cells(s, display, count):
    """The label each control cell shows: the highest-ranked label among its contacts."""
    ranks = np.zeros(count, np.uint8)
    np.maximum.at(ranks, s.ba, RANK[display])
    np.maximum.at(ranks, s.bb, RANK[display])
    return RANKED[ranks]


def incoming_water(s, down=None):
    """Whether each contact's incoming side carries any ocean."""
    import native_subduction
    owner = s.down if down is None else down
    if native_subduction.enabled(s):
        return native_subduction.edge_ocean_fraction(s, owner) > 1e-10
    owner = np.asarray(owner)
    below = np.where(owner == np.asarray(s.bp), np.asarray(s.ba), np.asarray(s.bb))
    return (np.asarray(s.crust)[below] == 0) & (owner >= 0)


def carries_material(s):
    """Whether either side of each contact carries buoyant crust.

    Measured on the material surface where that geometry exists: a margin can
    hold continental material inside a control cell the coarse crust flag still
    calls ocean, and a stalled trench is exactly that case.
    """
    import native_subduction
    if native_subduction.enabled(s):
        water_if = [native_subduction.edge_ocean_fraction(s, side) for side in (s.bp, s.bq)]
        return (water_if[0] < 1.-1e-10) | (water_if[1] < 1.-1e-10)
    crust = np.asarray(s.crust)
    return (crust[np.asarray(s.ba)] > 0) | (crust[np.asarray(s.bb)] > 0)


def codes(s):
    """Per-contact display codes: the 0-5 vocabulary of `s.bcode`, and 6 under version 2."""
    kinematic = np.asarray(s.bcode, np.uint8)
    selected = version(s)
    if not selected:
        return kinematic.copy()
    live = np.asarray(s.bl, float) > MINIMUM_LENGTH_KM
    display = np.where(live, TRANSFORM, NONE).astype(np.uint8)
    # Production decides divergence exactly; keep what it resolved.
    display[live & (kinematic == RIDGE)] = RIDGE
    display[live & (kinematic == RIFT)] = RIFT
    closing = normal_partition.convergence(s)
    # Continental material in a closing boundary is a collision, whether the
    # two sides are both continental or a buoyant plate has arrived and jammed
    # the margin. A stalled trench keeps no polarity (physics_profile.
    # stall_buoyant_incoming clears s.down), so polarity cannot identify it.
    display[live & closing & carries_material(s)] = COLLISION
    polarity = (np.asarray(s.down) == np.asarray(s.bp)) | (np.asarray(s.down) == np.asarray(s.bq))
    display[live & closing & polarity & incoming_water(s)] = SUBDUCTION
    if selected == CONVERGING_VERSION:
        # What is left is neither a running trench nor a collision. Where the gate calls the
        # motion convergent, it is not sliding either.
        display[(display == TRANSFORM) & np.isin(kinematic, (SUBDUCTION, COLLISION))] = CONVERGING
    return display


def lengths(s, display=None):
    """Boundary length in km for each label, from the same codes the map shows."""
    display = codes(s) if display is None else np.asarray(display, np.uint8)
    length = np.asarray(s.bl, float)
    labels = [('ridge', RIDGE), ('subduction', SUBDUCTION), ('transform', TRANSFORM),
              ('collision', COLLISION), ('rift', RIFT)]
    if version(s) == CONVERGING_VERSION:
        labels.append(('converging', CONVERGING))
    return {name: float(length[display == code].sum()) for name, code in labels}


def disagreement(s, display=None):
    """What the gate would have shown against what the mechanism is doing.

    Reported so the change is visible in every frame rather than silent.
    """
    display = codes(s) if display is None else np.asarray(display, np.uint8)
    kinematic = np.asarray(s.bcode, np.uint8)
    length = np.asarray(s.bl, float)
    moved = display != kinematic
    rows = {}
    for code, name in ((SUBDUCTION, 'subduction'), (COLLISION, 'collision'),
                       (TRANSFORM, 'transform'), (RIDGE, 'ridge'), (RIFT, 'rift'),
                       (CONVERGING, 'converging')):
        gained = moved & (display == code)
        if np.any(gained):
            rows[name] = dict(contacts=int(np.count_nonzero(gained)),
                              length_km=float(length[gained].sum()),
                              from_kinematic={str(int(c)): int(np.count_nonzero(gained & (kinematic == c)))
                                              for c in np.unique(kinematic[gained])})
    return dict(version=max(VERSION, version(s)), relabelled_contacts=int(np.count_nonzero(moved)),
                relabelled_length_km=float(length[moved].sum()), by_label=rows,
                policy='Labels follow the running mechanism; s.bcode remains the kinematic gate.')
