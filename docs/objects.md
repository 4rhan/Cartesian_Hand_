# Objects in simulation

`cartesian_hand/objects.py` turns a measured bottle into a MuJoCo scene that
`sim` can run a task against, and reports whether the cap came off.

```bash
python -m cartesian_hand.objects                            # list assets/objects/bottles.json
python -m cartesian_hand.objects water_500ml --view         # the bottle in the open jaws
python -m cartesian_hand.sim --task cap --obj water_500ml   # run it, headless
python -m cartesian_hand.sim --task cap --obj water_500ml --viewer   # watch it, real time
```

A run ends the tick the cap is lifted 10 mm off the neck, with one line from
the object:

```
[obj] water_500ml: turned 1.01 of 1.00 turns, released at 9.8s, cap lifted 11.0 mm off the neck -> OPENED
```

The closing half of `cap` isn't modelled (see the thread, below), so it is
skipped. Add `--full-cycle` to run it anyway.

To see a bottle, use `--view` or `--viewer` as above, or open its generated
scene, `assets/cartesian_hand/_scene_<name>.xml`. `scene.xml` itself is the
empty scene with no object in it.

### Many bottles from one measurement

```bash
python -m cartesian_hand.objects water_500ml --sweep-n 20             # +/-20% variants
python -m cartesian_hand.objects water_500ml --sweep-n 20 --spread 0.3 --seed 1
```

This varies sizes, thread pitch, turns, masses and friction around the
measurement (resisting torque over a factor of 2 either way), runs `cap` on
each variant, and prints how many opened. About 17 s per variant on CPU.

## Adding a real bottle

Add an entry to `assets/objects/bottles.json`. Every number is a measurement
of the physical object, in millimetres, grams and newton-metres:

| Field | How to measure |
|---|---|
| `body_radius` | Calipers across the body where the base jaw will hold it, halved. Square bottle: half the side, and set `"shape": "square"`. |
| `body_height` | Base to shoulder (where the body narrows into the neck). |
| `neck_radius`, `neck_height` | Across the neck just below the cap, halved; shoulder to the cap's lower edge. |
| `cap_radius`, `cap_height` | Across the cap's grip surface, halved; its height. |
| `mass_g`, `cap_mass_g` | Kitchen scale: bottle with its contents but without the cap; the cap alone. |
| `thread_pitch` | Across N thread crests on the neck, divided by N. Or: how far the cap rises in one turn. |
| `turns_to_open` | Turns from closed until the cap lifts off freely. Count them by hand. |
| `break_torque` | Resistance while turning, N*m. 0.02 to 0.05 for an opened cap. A sealed one needs far more, so measure it with a torque screwdriver or start low. |
| `friction` | 1.0 for a ridged plastic cap. Lower for smooth or wet ones. |

`python -m cartesian_hand.objects` checks each entry against the hand. A radius
over 48 mm does not fit the jaws, and a bottle shorter than ~50 mm leaves the
base jaw nothing to hold. Each problem is reported with the field to fix.

### Looks: a scanned mesh (optional)

Collision always uses the primitives above. A twist rolls the cap between two
flat pads, and only a true cylinder rolls smoothly; a scanned mesh has facets
the cap would bump over on every stroke. A scan is still worth having for
camera images (VLA data). Scan the object with a phone app (Polycam,
Scaniverse, RealityScan), export OBJ or STL, put it under `assets/objects/`,
and name it in the entry:

```json
"visual_mesh": "my_bottle/body.obj",
"cap_visual_mesh": "my_bottle/cap.obj",
"mesh_scale": 0.001
```

The body mesh has its origin at the base's centre and the cap mesh at its lower
edge's centre, both with z up. `mesh_scale` converts the mesh's units to
metres: 0.001 for millimetres. Check the result with `--view`.

## How the scene is built

- **Placement.** The bottle stands at x = 100 mm between the finger pads, which
  every finger position in `cap`'s stroke still covers. The cap's lower edge is
  1 mm under the aux pads at `cap_offset` (default 15). The table goes wherever
  that leaves the bottle's base, since the hand is fixed to the world as it is
  on its mount.
- **The thread** is two joints, not geometry. The cap is welded to a carriage
  on a hinge (with `break_torque` as constant friction) and a slide coupled to
  it at `thread_pitch` per turn. Past `turns_to_open` the weld and coupling
  switch off and the cap is a loose body sitting on the neck. Re-threading is
  not modelled, so the closing half of `cap` turns a loose cap. These scenes
  measure *opening*.
- **Task settings.** `--obj` sets `cap_offset` and asks for `turns_to_open +
  0.25` turns, the slip margin you would give `num_revs_up` on the bench.

## Sim-only adjustments, and why

Three settings differ from the empty-hand model. Each was measured against a
failure:

- **Object contacts are stiff** (`objects.CONTACT`). At MuJoCo's defaults a
  2.5 g cap squeezed at ~12 N sank 4 mm into the pads, so the probe read an
  11 mm radius on a 15 mm cap and sized every stroke for the wrong cap.
- **Thread constraints are stiff.** At the default the cap slid 5.7 mm down the
  neck under its own weight before anything touched it.
- **Finger gains are raised to kp=3000** (`objects.FINGER_KP`), in object
  scenes only. The generated model's kp=150 could not turn a cap against
  0.02 N*m: 0.26 turns, then the stroke timed out. This belongs in
  `assets/cartesian_hand/source/kinematics.json` once the real servo's
  stiffness is measured.

**Grip force is a force cap, as on the servo.** Every tick `sim.py` turns
each joint's effort (`Motions` torque / 1000, or `Action.effort_limit`) into
that actuator's force limit, scaled by the model's full-scale +/-87 N. So
`squeeze_torque` matters: on `water_500ml` the default 80 opens the cap and
20 slips at 0.98 turns. The 87 N is inherited from the sibling hand and not
yet measured for this one, so bench torques transfer as ratios, not as
newtons. CPU backend only; `--warp` still ignores effort.

Try it with `--set`, which overrides any field of the task's `Config`:

```bash
python -m cartesian_hand.sim --task cap --obj water_500ml --set squeeze_torque=20
```
