"""Real objects in the sim: a measured bottle becomes a scene the hand can open.

    python -m cartesian_hand.objects                       # list the catalog
    python -m cartesian_hand.objects water_500ml --view    # look at it in the jaws
    python -m cartesian_hand.sim --task cap --obj water_500ml --viewer
    python -m cartesian_hand.objects water_500ml --sweep-n 20  # 20 random variants

An object is a set of caliper measurements in `assets/objects/bottles.json`,
not a mesh. Collision is built from primitives (cylinder or box body, cylinder
neck, cylinder cap) because a twist rolls the cap between two flat pads, and a
primitive cylinder is the one shape whose contact against a pad is smooth: a
scanned or CoACD mesh has facets the cap would bump over every stroke. A scan
is still welcome as *looks* -- `visual_mesh` and `cap_visual_mesh` attach one
with collision off, which is what a camera policy needs and the physics does
not.

The thread
----------
Not modelled as geometry -- a helical contact is slow and unstable, and every
sim-to-real cap-twisting result to date fakes it. The cap is welded to a small
`thread` carriage that sits on the bottle's neck on two joints:

    cap_turn    hinge about the bottle axis, `frictionloss = break_torque`
    cap_rise    slide along it, held to `pitch * turn / 2pi` by an equality

so the cap rises one pitch per turn and resists turning with a constant
torque. Once `cap_turn` passes `turns_to_open`, `ThreadMonitor` switches the
weld and the coupling off and the cap becomes a free body: the hand has to
hold it or it falls. There is no re-engagement, so the closing half of
`tasks/cap.py` turns a loose cap -- this measures *opening* only.

Placement
---------
The bottle stands on its axis at `OBJECT_X_MM` between the finger pads, which
is the one x every finger position in `cap`'s 0..45 mm stroke still covers.
Heights follow from the hand, not from the bottle: the cap's bottom sits
`AUX_PAD_BOTTOM_MM - 1` above the aux pads' lowest edge at `cap_offset`, so the
aux pads close on the cap and not on the shoulder below it. The table is moved
down to wherever that leaves the bottle's base; the hand is fixed to the world,
as it is on its mount, so it does not care where the table is.

A `Placement` moves the bottle off that nominal pose the way a person setting
it down does: a few mm off centre, turned, slightly tilted. Tilt is the table's,
not the bottle's: a bottle tilted on a level table just rocks back flat, so the
table under it is tilted instead and the bottle stands square on it.
"""
import contextlib
import dataclasses
import io
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Annotated

import mujoco
import numpy as np
import tyro

from .mjcf import MM_PER_M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(ROOT, "assets", "objects", "bottles.json")
SCENE = os.path.join(ROOT, "assets", "cartesian_hand", "scene.xml")
GENERATED = os.path.join(ROOT, "assets", "cartesian_hand", "_scene_{}.xml")
"""Generated scenes sit beside `scene.xml`, not under `assets/objects/`:
MuJoCo resolves the hand's `meshdir` against the top-level file's directory."""

OBJECT_X_MM = 100.0
"""Bottle axis, along the fingers. Pads span x = 34..129 mm at finger 0 and move
with the finger, so x = 100 is under both pads for the whole 0..45 mm stroke."""

BASE_PAD_MM = (-16.0, 16.0)
"""z extent of the base fingers' pads. Fixed: the base jaw does not ride z."""

AUX_PAD_BOTTOM_MM = 16.0
"""Lowest edge of the aux pads at z = 0; it rides z one-for-one."""

JAW_TRAVEL_MM = 50.0
"""`config.STANDARD_TRAVEL` for both jaws: the widest half-opening there is."""

FINGER_KP, FINGER_KV = 3000.0, 15.0
"""Finger position gains in object scenes, N/m and N*s/m -- the z stage's.

The generated model gives the fingers kp=150, a placeholder that nothing in an
empty scene could expose: turning a cap against 0.02 N*m takes ~1.3 N at the
pads, which at 150 N/m is 9 mm of lag, so the driving finger stalled short and
the cap dragged the other one past its goal (measured: 0.26 turns, then the
stroke timed out). A position servo holds far stiffer than that. At 1000, 3000
and 8000 the same stroke turns 0.91, 0.94 and 0.99 of a turn. The real fix
belongs in `assets/cartesian_hand/source/kinematics.json`; this keeps the
generated hand untouched until the gains there are measured."""

OPEN_MARGIN_TURNS = 0.25
"""Extra turns `teacher_config` asks `cap` for beyond `turns_to_open`, for slip --
the same margin set by hand on the bench through `num_revs_up`."""

IMAGE_SIZE = 224
"""Policy camera resolution, square: pi0.5's image input (openpi resizes every
view to 224 x 224, so rendering larger only costs time)."""

CAMERAS = {
    # name: (parent body, position mm, look-at point mm), both in the parent's
    # frame at the start pose. The bottle's cap sits at x = 100, z ~ 30-45.
    "base": ("world", (320.0, -70.0, 150.0), (100.0, 0.0, 25.0)),
    "wrist": ("bridge", (200.0, 0.0, 95.0), (100.0, 0.0, 30.0)),
}
"""The two policy views. `base` is fixed in the world, a three-quarter view of
the bottle between the jaws -- openpi's `base_0_rgb`. `wrist` rides the z
stage, so it moves with the aux jaw like a camera bolted to it would, looking
back and down at the cap from past the fingertips -- `left_wrist_0_rgb`."""


def look_at(pos_mm, target_mm) -> list[float]:
    """MuJoCo `xyaxes` for a camera at `pos_mm` looking at `target_mm`, z up.
    A MuJoCo camera looks along its own -z, with +y up in the image."""
    f = np.subtract(target_mm, pos_mm, dtype=float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return [*x, *y]


# Hand collision geoms are contype=1, conaffinity=0, so they touch only what
# accepts type 1. Objects accept it, and are type 1 so the table accepts them.
#
# Stiff, and weighted to win the per-pair average (`solmix`): MuJoCo's soft
# contact scales with the bodies' mass, so at the default a 2.5 g cap squeezed
# at ~12 N sank 4 mm into the pads, `cap`'s probe measured an 11 mm radius on a
# 15 mm cap, and every stroke after it was sized for the wrong cap. These
# settings stop both jaws within 0.1 mm of the true radius.
CONTACT = dict(contype=1, conaffinity=1, condim=4,
               solref=[0.002, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
               solmix=100.0)


@dataclass(frozen=True)
class Placement:
    """Where the bottle stands, relative to the nominal pose. mm and degrees."""

    dx_mm: float = 0.0
    """Along the fingers (+x)."""
    dy_mm: float = 0.0
    """Across the jaws (+y)."""
    yaw_deg: float = 0.0
    """About the bottle's own axis. Shows on a square body and on a scan."""
    tilt_deg: float = 0.0
    """How far the table, and so the bottle, leans from vertical."""
    tilt_dir_deg: float = 0.0
    """Which way the top leans, measured from +x toward +y."""

    def rotation(self) -> tuple[list[float], list[float]]:
        """(bottle quat, table quat), MuJoCo's w-x-y-z."""
        tilt = np.zeros(4)
        d = math.radians(self.tilt_dir_deg)
        mujoco.mju_axisAngle2Quat(tilt, [-math.sin(d), math.cos(d), 0.0],
                                  math.radians(self.tilt_deg))
        yaw = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw, [0.0, 0.0, 1.0], math.radians(self.yaw_deg))
        both = np.zeros(4)
        mujoco.mju_mulQuat(both, tilt, yaw)
        return both.tolist(), tilt.tolist()


PLACEMENT_RANGE = dict(offset_mm=3.0, tilt_deg=2.0)
"""What `sample_placement` draws within: +/- offset on x and y, any yaw, and a
tilt up to this many degrees in any direction."""


def sample_placement(rng: random.Random, offset_mm: float = PLACEMENT_RANGE["offset_mm"],
                     tilt_deg: float = PLACEMENT_RANGE["tilt_deg"]) -> Placement:
    return Placement(dx_mm=rng.uniform(-offset_mm, offset_mm),
                     dy_mm=rng.uniform(-offset_mm, offset_mm),
                     yaw_deg=rng.uniform(0.0, 360.0),
                     tilt_deg=rng.uniform(0.0, tilt_deg),
                     tilt_dir_deg=rng.uniform(0.0, 360.0))


@dataclass(frozen=True)
class Bottle:
    """One real object, as measured. Millimetres, grams, newton-metres."""

    name: str
    body_radius: float
    """Half the width the base jaw closes on. A square bottle: half its side."""
    body_height: float
    """Base to shoulder."""
    neck_radius: float
    neck_height: float
    """Shoulder to the cap's lower edge."""
    cap_radius: float
    cap_height: float
    mass_g: float
    """Bottle including contents, without the cap."""
    cap_mass_g: float
    thread_pitch: float = 3.0
    """Rise per turn. Calipers across N thread crests, divided by N."""
    turns_to_open: float = 1.0
    """Turns from closed until the cap lifts off freely."""
    break_torque: float = 0.02
    """Constant resisting torque while turning, N*m (a sealed cap needs more)."""
    friction: float = 1.0
    shape: str = "round"
    """"round" or "square"."""
    cap_offset: float = 15.0
    """z stage height that puts the aux pads on the cap -- passed to `cap`."""
    body_rgba: tuple = (0.6, 0.8, 1.0, 0.6)
    cap_rgba: tuple = (0.1, 0.35, 0.85, 1.0)
    visual_mesh: str | None = None
    """Optional scan of the bottle, OBJ/STL, origin at the base centre, z up.
    Relative paths resolve against `assets/objects/`."""
    cap_visual_mesh: str | None = None
    """Optional scan of the cap, origin at its lower edge's centre, z up."""
    mesh_scale: float = 0.001
    """Mesh units to metres. 0.001 for a scan exported in millimetres."""

    @property
    def cap_bottom_mm(self) -> float:
        return self.cap_offset + AUX_PAD_BOTTOM_MM - 1.0

    @property
    def base_z_mm(self) -> float:
        """Where the bottle's base and the table end up."""
        return self.cap_bottom_mm - self.neck_height - self.body_height

    def check(self) -> None:
        """Refuse an object the hand cannot hold, saying which measurement."""
        widest = max(self.body_radius, self.cap_radius)
        if widest + 2.0 > JAW_TRAVEL_MM:
            raise ValueError(f"{self.name}: radius {widest} mm does not fit a "
                             f"jaw that opens to {JAW_TRAVEL_MM} mm")
        if self.neck_radius > self.cap_radius:
            raise ValueError(f"{self.name}: neck wider than cap")
        if self.base_z_mm > BASE_PAD_MM[0] - 1.0:
            raise ValueError(
                f"{self.name}: too short -- base at {self.base_z_mm:.1f} mm is "
                f"above the base pads' lower edge {BASE_PAD_MM[0]} mm; lower "
                f"cap_offset or check body_height")
        shoulder = self.cap_bottom_mm - self.neck_height
        if min(shoulder, BASE_PAD_MM[1]) - BASE_PAD_MM[0] < 10.0:
            raise ValueError(f"{self.name}: under 10 mm of body in the base "
                             f"jaw's reach (shoulder at {shoulder:.1f} mm)")
        if self.shape not in ("round", "square"):
            raise ValueError(f"{self.name}: shape must be round or square")


def catalog(path: str = CATALOG) -> dict[str, Bottle]:
    with open(path) as f:
        raw = json.load(f)
    fields = {f.name for f in dataclasses.fields(Bottle)}
    out = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        unknown = set(entry) - fields
        if unknown:
            raise KeyError(f"{name}: unknown field(s) {sorted(unknown)}")
        out[name] = Bottle(name=name, **{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in entry.items()})
    return out


def load(name: str) -> Bottle:
    objects = catalog()
    if name not in objects:
        raise KeyError(f"unknown object {name!r}; {CATALOG} has "
                       f"{sorted(objects)}")
    bottle = objects[name]
    bottle.check()
    return bottle


def _mesh(spec: mujoco.MjSpec, name: str, file: str, scale: float) -> str:
    path = file if os.path.isabs(file) else os.path.join(ROOT, "assets", "objects", file)
    spec.add_mesh(name=name, file=path, scale=[scale] * 3)
    return name


def build_spec(b: Bottle, place: Placement = Placement()) -> mujoco.MjSpec:
    """`scene.xml` plus this bottle, its cap, the thread, and a start keyframe."""
    m = 1.0 / MM_PER_M
    spec = mujoco.MjSpec.from_file(SCENE)
    world = spec.worldbody
    quat, table_quat = place.rotation()
    base = np.array([OBJECT_X_MM + place.dx_mm, place.dy_mm, b.base_z_mm]) * m
    floor = spec.geom("floor")
    floor.pos, floor.quat = base.tolist(), table_quat

    bottle = world.add_body(name="bottle", pos=base.tolist(), quat=quat)
    bottle.add_freejoint(name="bottle_free")
    body_size = ([b.body_radius * m, b.body_height / 2 * m, 0.0]
                 if b.shape == "round" else
                 [b.body_radius * m, b.body_radius * m, b.body_height / 2 * m])
    bottle.add_geom(name="bottle_body",
                    type=(mujoco.mjtGeom.mjGEOM_CYLINDER if b.shape == "round"
                          else mujoco.mjtGeom.mjGEOM_BOX),
                    size=body_size, pos=[0, 0, b.body_height / 2 * m],
                    mass=b.mass_g / 1000.0, rgba=list(b.body_rgba),
                    friction=[b.friction, 0.005, 0.0001], **CONTACT)
    top = (b.body_height + b.neck_height) * m
    bottle.add_geom(name="bottle_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[b.neck_radius * m, b.neck_height / 2 * m, 0.0],
                    pos=[0, 0, (b.body_height + b.neck_height / 2) * m],
                    mass=1e-3, rgba=list(b.body_rgba), **CONTACT)

    # The carriage the cap is welded to. Light but not massless: a near-zero
    # inertia on a constrained joint is what makes a solver ring. Its mass is a
    # collision-free geom rather than an explicit inertial because
    # `MjSpec.to_xml` drops the inertial of a body that has no geoms.
    thread = bottle.add_body(name="thread", pos=[0, 0, top])
    thread.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0],
                    mass=1e-3, contype=0, conaffinity=0, group=4)
    thread.add_joint(name="cap_turn", type=mujoco.mjtJoint.mjJNT_HINGE,
                     axis=[0, 0, 1], frictionloss=b.break_torque,
                     damping=1e-4, armature=1e-5)
    thread.add_joint(name="cap_rise", type=mujoco.mjtJoint.mjJNT_SLIDE,
                     axis=[0, 0, 1], damping=1e-3, armature=1e-4,
                     range=[-0.005, 0.1])

    # The cap is its own free body so that, released, it is simply loose.
    # Placed where the carriage is, in the bottle's tilted frame: the weld
    # below assumes the two start coincident.
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, [0.0, 0.0, top], quat)
    cap = world.add_body(name="cap", pos=(base + up).tolist(), quat=quat)
    cap.add_freejoint(name="cap_free")
    cap.add_geom(name="cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[b.cap_radius * m, b.cap_height / 2 * m, 0.0],
                 pos=[0, 0, b.cap_height / 2 * m], mass=b.cap_mass_g / 1000.0,
                 rgba=list(b.cap_rgba), friction=[b.friction, 0.005, 0.0001],
                 **CONTACT)

    for parent, mesh in ((bottle, b.visual_mesh), (cap, b.cap_visual_mesh)):
        if mesh:
            name = _mesh(spec, f"{parent.name}_scan", mesh, b.mesh_scale)
            parent.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                            contype=0, conaffinity=0, group=2, mass=0)
    if b.visual_mesh:
        spec.geom("bottle_body").group = 3   # hide the primitive behind the scan
        spec.geom("bottle_neck").group = 3
    if b.cap_visual_mesh:
        spec.geom("cap").group = 3

    # Both constraints stiffer than MuJoCo's default (solref 0.02): a soft
    # constraint sags under gravity by roughly g * timeconst^2, which at the
    # default let the cap slide 5.7 mm down the neck before anything touched
    # it. 0.001 is the floor (twice the 0.5 ms timestep); with this solimp the
    # sag is 0.04 mm and the run stays stable.
    stiff = dict(solref=[0.001, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])
    # Right-hand thread: turning +z (anticlockwise seen from above) unscrews.
    spec.add_equality(name="thread_coupling", type=mujoco.mjtEq.mjEQ_JOINT,
                      objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      name1="cap_rise", name2="cap_turn",
                      data=[0, b.thread_pitch * m / (2 * math.pi), 0, 0, 0]
                      + [0] * 6, **stiff)
    # Explicit identity relpose: cap and carriage start coincident.
    spec.add_equality(name="cap_weld", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="cap", name2="thread",
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], **stiff)
    # No bottle/cap contact exclusion: the cap sits on top of the neck rather
    # than around it, so the two only touch face to face, and a released cap
    # has to rest on the neck as a real one does -- excluded, it fell straight
    # through the bottle the moment `cap` opened the jaw to re-grip.

    for name, (parent, pos, target) in CAMERAS.items():
        body = world if parent == "world" else spec.body(parent)
        body.add_camera(name=name, pos=[v * m for v in pos],
                        xyaxes=look_at(pos, target), fovy=50.0)

    for actuator in spec.actuators:
        if "finger" in actuator.name:
            actuator.gainprm[0] = FINGER_KP
            actuator.biasprm[1] = -FINGER_KP
            actuator.biasprm[2] = -FINGER_KV

    # Start with both jaws open past the widest part, so nothing begins
    # interpenetrated: qpos 0 is the closed jaw.
    model = spec.compile()
    qpos = model.qpos0.copy()
    opening = min(max(b.body_radius, b.cap_radius) + 6.0,
                  JAW_TRAVEL_MM - 0.5) * m
    for j in ("left_up_y", "right_up_y", "left_down_y", "right_down_y"):
        qpos[model.jnt_qposadr[model.joint(j).id]] = opening
    spec.add_key(name="home", qpos=qpos.tolist())
    return spec


def randomize_visuals(model: mujoco.MjModel, rng: random.Random) -> None:
    """Redraw lighting and colours in a compiled scene, in place.

    Only fields the renderer reads: light intensity and placement, the
    headlight, and the rgba of the bottle, cap, table and hand. Nothing here
    reaches the physics, so an episode's outcome does not depend on it -- a
    policy that changes its answer when the cap turns red has learned the
    colour, not the cap.
    """
    for i in range(model.nlight):
        model.light_diffuse[i] *= rng.uniform(0.4, 1.3)
        model.light_pos[i] += [rng.uniform(-0.15, 0.15) for _ in range(3)]
        d = model.light_dir[i] + [rng.uniform(-0.3, 0.3) for _ in range(3)]
        model.light_dir[i] = d / np.linalg.norm(d)
    head = model.vis.headlight
    head.diffuse[:] = np.array(head.diffuse) * rng.uniform(0.5, 1.3)
    head.ambient[:] = np.array(head.ambient) * rng.uniform(0.5, 1.5)

    def colour() -> list[float]:
        return [rng.random() for _ in range(3)]
    model.geom("cap").rgba[:3] = colour()
    for name in ("bottle_body", "bottle_neck"):
        model.geom(name).rgba = [*colour(), rng.uniform(0.35, 0.95)]
    model.material("groundplane").rgba[:3] = [rng.uniform(0.3, 1.0)
                                              for _ in range(3)]
    for i in range(model.nmat):
        if model.material(i).name != "groundplane":
            model.mat_rgba[i, :3] = np.clip(
                model.mat_rgba[i, :3] * rng.uniform(0.7, 1.3), 0.0, 1.0)
    for i in np.flatnonzero(model.geom_matid < 0):
        model.geom_rgba[i, :3] = np.clip(
            model.geom_rgba[i, :3] * rng.uniform(0.85, 1.15), 0.0, 1.0)


def write_scene(b: Bottle, place: Placement = Placement()) -> str:
    """Write the scene for `b` and return its path, ready for `--xml` or the
    MuJoCo viewer."""
    spec = build_spec(b, place)
    path = GENERATED.format(b.name)
    with open(path, "w") as f:
        f.write(spec.to_xml())
    return path


class ThreadMonitor:
    """Per-tick hook for `sim.run`: releases the cap, records what happened.

    Passed as `external_signal`, which `drive` calls once a tick with the model
    and data. It returns None -- it reports no contact, so tasks behave exactly
    as they would without it -- and has one side effect, the release.
    """

    def __init__(self, bottle: Bottle):
        self.bottle = bottle
        self.released_at_s: float | None = None
        self.max_turns = 0.0
        self.max_lift_mm = 0.0
        """Highest the cap got above the neck after release. Recorded rather
        than read at the end because `cap` then puts the cap back, and with no
        re-threading here it ends wherever a loose cap falls."""
        self._ids = None

    def _resolve(self, model: mujoco.MjModel) -> None:
        jid = lambda n: model.joint(n).id
        self._turn = model.jnt_qposadr[jid("cap_turn")]
        self._eq = [model.equality(n).id for n in ("thread_coupling", "cap_weld")]
        self._cap = model.body("cap").id
        self._bottle = model.body("bottle").id
        self._ids = True

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._ids is None:
            self._resolve(model)
            self.model, self.data = model, data
        turns = data.qpos[self._turn] / (2 * math.pi)
        self.max_turns = max(self.max_turns, turns)
        if self.released_at_s is None and turns >= self.bottle.turns_to_open:
            data.eq_active[self._eq] = 0
            self.released_at_s = float(data.time)
        if self.released_at_s is not None:
            self.max_lift_mm = max(self.max_lift_mm, self._lift_mm(data))
        return None

    def _lift_mm(self, data: mujoco.MjData) -> float:
        """Cap's lower edge above the neck's top, along the bottle's axis.

        Against the bottle, not the `thread` carriage: released, the carriage
        is free to slide down its joint, and measured against it a cap the jaws
        merely held still read as 10 mm lifted the tick the weld let go."""
        top = (self.bottle.body_height + self.bottle.neck_height) / MM_PER_M
        axis = data.xmat[self._bottle].reshape(3, 3)[:, 2]
        rel = data.xpos[self._cap] - data.xpos[self._bottle]
        return float(rel @ axis - top) * MM_PER_M

    OPENED_LIFT_MM = 10.0
    """Lift above the neck, after release, that counts as the cap removed."""

    @property
    def opened(self) -> bool:
        """Released and lifted `OPENED_LIFT_MM` off the neck at some point."""
        return (self.released_at_s is not None
                and self.max_lift_mm >= self.OPENED_LIFT_MM)

    def summary(self) -> str:
        if self._ids is None:
            return "[obj] never ran"
        if self.released_at_s is None:
            return (f"[obj] {self.bottle.name}: turned {self.max_turns:.2f} of "
                    f"{self.bottle.turns_to_open:.2f} turns, thread still engaged "
                    f"-> NOT OPENED")
        opened = self.opened
        return (f"[obj] {self.bottle.name}: turned {self.max_turns:.2f} of "
                f"{self.bottle.turns_to_open:.2f} turns, released at "
                f"{self.released_at_s:.1f}s, cap lifted {self.max_lift_mm:.1f} mm "
                f"off the neck -> {'OPENED' if opened else 'NOT OPENED'}")


def teacher_config(task_cfg: object | None, b: Bottle) -> object | None:
    """The task's `Config` set for this bottle: its `cap_offset`, and enough
    opening turns to free the cap. None for a task with neither field.

    **Privileged: the scripted teacher's only.** Both numbers are the answer --
    where the cap is and how far it must turn -- read off the measurement
    rather than found by the hand. The teacher may have them because it is
    what generates demonstrations; a policy under test must not, or a sim
    success says nothing about the bench. `env.CapEnv`'s `obs` is what such a
    policy gets instead."""
    if task_cfg is None:
        return None
    changes = {}
    if hasattr(task_cfg, "cap_offset"):
        changes["cap_offset"] = b.cap_offset
    if hasattr(task_cfg, "num_revs_up"):
        changes["num_revs_up"] = b.turns_to_open + OPEN_MARGIN_TURNS
    return dataclasses.replace(task_cfg, **changes) if changes else None


VARIED = {
    # field: how it is drawn around the measured value
    "body_radius": "scale", "body_height": "scale",
    "neck_radius": "scale", "neck_height": "scale",
    "cap_radius": "scale", "cap_height": "scale",
    "mass_g": "scale", "cap_mass_g": "scale",
    "thread_pitch": "scale", "turns_to_open": "scale",
    "break_torque": "log", "friction": "scale",
}
"""What `vary` changes. "scale" is uniform in +/- `spread` of the measured
value; "log" is log-uniform over a factor (`vary`'s `log_factor`, default 2)
either way, because a cap's resisting torque is known to its order of
magnitude, not to 20%."""


def vary(b: Bottle, rng: random.Random, spread: float = 0.2,
         name: str | None = None, log_factor: float = 2.0) -> Bottle:
    """A random bottle around the measured one `b` that the hand can hold.

    `log_factor` bounds the "log" fields: drawn over that factor either way.

    Redraws until `check` passes, so a variant too wide for the jaws or too
    short for the base jaw is never handed out -- the sweep measures the task,
    not the catalog's edge cases.
    """
    for _ in range(1000):
        changes = {}
        for field, how in VARIED.items():
            value = getattr(b, field)
            changes[field] = (value * log_factor ** rng.uniform(-1.0, 1.0)
                              if how == "log" else
                              value * rng.uniform(1.0 - spread, 1.0 + spread))
        changes["neck_radius"] = min(changes["neck_radius"],
                                     changes["cap_radius"] - 1.0)
        candidate = dataclasses.replace(b, name=name or f"{b.name}_var",
                                        **changes)
        try:
            candidate.check()
        except ValueError:
            continue
        return candidate
    raise ValueError(f"{b.name}: no valid variant in 1000 draws at "
                     f"spread {spread}")


def evaluate(b: Bottle, task: str = "cap",
             hand: str | None = None) -> tuple[bool, str]:
    """Run `task` on `b` in CPU MuJoCo. Returns (cap opened, one-line verdict).

    The task's own row log is swallowed; a task that raises still gets its
    verdict, since a run that fails while closing may well have opened first.
    """
    import torch
    from . import sim, tasks
    from .config import DEFAULT_HAND, get_hand

    cfg = get_hand(hand or DEFAULT_HAND)
    monitor = ThreadMonitor(b)
    kwargs = {}
    task_cfg = teacher_config(tasks.config(task), b)
    if task_cfg is not None:
        kwargs["cfg"] = task_cfg
    controller = tasks.make(task, cfg, torch.zeros(1, cfg.n_dof), **kwargs)
    log = io.StringIO()
    error = ""
    with contextlib.redirect_stdout(log):
        try:
            sim.run(controller, cfg, xml=write_scene(b),
                    external_signal=monitor, verbose=False)
        except RuntimeError as e:
            failed = [l.strip() for l in log.getvalue().splitlines()
                      if "FAILED" in l]
            error = f"  [task: {failed[0] if failed else e}]"
    return monitor.opened, monitor.summary() + error


def sweep(b: Bottle, n: int, spread: float = 0.2, seed: int = 0,
          task: str = "cap") -> float:
    """Run `task` on `n` variants of `b`; print each and the success rate."""
    rng = random.Random(seed)
    wins = 0
    for i in range(n):
        v = vary(b, rng, spread, name=f"{b.name}_v{i:03d}")
        opened, verdict = evaluate(v, task)
        wins += opened
        print(f"  body r{v.body_radius:4.1f}  cap r{v.cap_radius:4.1f} "
              f"h{v.cap_height:4.1f}  pitch {v.thread_pitch:3.1f}  "
              f"turns {v.turns_to_open:4.2f}  torque {v.break_torque:.3f}  "
              f"mu {v.friction:.2f}  {'OPENED' if opened else 'failed'}")
        if not opened:
            print(f"      {verdict}")
        os.remove(GENERATED.format(v.name))
    rate = wins / n
    print(f"[sweep] {b.name}: {wins}/{n} opened ({rate:.0%}), spread "
          f"+/-{spread:.0%}, seed {seed}")
    return rate


def main(name: Annotated[str | None, tyro.conf.Positional] = None,
         view: bool = False, sweep_n: int = 0, spread: float = 0.2,
         seed: int = 0, placement_seed: int | None = None) -> None:
    """List the object catalog, or write (and optionally view) one object's scene.

    Args:
        name: an entry in assets/objects/bottles.json. Omit to list them.
        view: open the MuJoCo viewer on the scene, jaws open at the start pose.
        sweep_n: run `cap` on this many random variants of the object and
            report how many opened. 0 runs none.
        spread: how far a variant's sizes, pitch, turns, masses and friction
            stray from the measurement, as a fraction (0.2 = +/-20%).
        seed: variants are reproducible from this.
        placement_seed: stand the bottle at a random `Placement` drawn from
            this seed (offset, yaw, tilt) instead of the nominal pose.
    """
    if name is None:
        for n, b in catalog().items():
            try:
                b.check()
                status = "ok"
            except ValueError as e:
                status = f"INVALID: {e}"
            print(f"  {n:16s} body r{b.body_radius:g}x{b.body_height:g}  "
                  f"cap r{b.cap_radius:g}x{b.cap_height:g}  {status}")
        return
    b = load(name)
    if sweep_n:
        sweep(b, sweep_n, spread, seed)
        return
    place = (Placement() if placement_seed is None else
             sample_placement(random.Random(placement_seed)))
    path = write_scene(b, place)
    print(f"wrote {path}")
    if placement_seed is not None:
        print(f"  placement {place}")
    print(f"  cap_offset {b.cap_offset:g} mm, table at z {b.base_z_mm:.1f} mm")
    if view:
        from mujoco import viewer as mj_viewer
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = [data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                        for i in range(model.nu)]
        mj_viewer.launch(model, data)


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.objects")
