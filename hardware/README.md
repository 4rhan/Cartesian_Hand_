# Cartesian Hand

A 4-finger cartesian gripper driven by seven Feetech HLS3915M serial bus servos. This page lists everything needed to build it.

<img src="../media/design-cad.jpg" alt="Cartesian Hand CAD" width="380">

To build the hand: print the parts, make two wire splices, give each servo an ID, assemble, then verify and zero it in software. The steps are in that order under [Assembly](#assembly), ending with [Verify and zero](#5-verify-and-zero).

## Tools Needed

- 3D printer (see [Print Settings](#print-settings-for-all-3dp-parts))
- Soldering iron and solder
- Hot air gun, for shrinking heat shrink
- Wire cutters and wire strippers
- Ruler, for measuring wire lengths
- Pliers, flush snippers and small pointy tools, for removing supports
- Screwdriver and/or Allen (hex) keys, whichever matches your screw heads, for M2 and M2.5 screws (and M3 for the optional Franka mount)
- Tweezers, for holding small parts and wires
- Multimeter, for checking the wires you make
- Tape and marker, for labelling servos
- Sandpaper (optional), for smoothing the rails if the parts do not slide smoothly
- A computer with a USB port, running Linux (the software does not build on Windows)

## Bill of Materials

### Electronics

| Qty | Part | Notes | Link |
|----:|------|-------|------|
| 1 | Serial bus servo driver board | Controls all servos over one bus | [Amazon](https://www.amazon.com/Serial-Integrates-Control-Circuit-Supports/dp/B0D1R4SGFS) |
| 7 | Feetech HLS3915M servo | Sold as 2-packs (buy 4 packs, 1 spare). Each servo includes a 150mm JST cable | [Amazon](https://www.amazon.com/2-Pack-Feetech-HLS3915M-Dual-Shaft-Coreless/dp/B0GVY2Y3C9) |
| 1 | Pre-made JST 1.25mm 3-pin wires, 200mm (pack) | Used for the Y-splice cables (6 wires needed) | [Amazon](https://www.amazon.com/dp/B0FS155VV6) |
| 1 | Heat shrink tubing assortment | For insulating the Y-splice joints | [Amazon](https://www.amazon.com/Eventronic-Heat-Shrink-Tubing-Kit-3/dp/B0BVVMCY86) |

### Hardware and Materials

| Qty | Part | Notes | Link |
|----:|------|-------|------|
| 1 | Bambu PLA Basic filament | All printed parts | [Bambu](https://us.store.bambulab.com/products/pla-basic-filament?id=40988815556744) |
| 1 | Bambu Support for PLA filament | Support interface layer only | [Bambu](https://us.store.bambulab.com/products/support-for-pla-new) |
| 20 | M2 x 6 mm screw | | [Amazon (assortment kit)](https://www.amazon.com/Fgruh-1200PCS-Assortment-Washers-Assorted/dp/B0FGV5K8BT) |
| 7 | M2.5 x 10 mm screw | | [Amazon (assortment kit)](https://www.amazon.com/Fgruh-1200PCS-Assortment-Washers-Assorted/dp/B0FGV5K8BT) |
| 1 | Grip tape | Cut to fit the finger tips | [Amazon](https://www.amazon.com/gp/product/B0093CQQNQ) |

### 3D-Printed Parts

| Qty | Part | Notes |
|----:|------|-------|
| 1 | Base gripper rail | |
| 1 | Auxiliary gripper rail | |
| 4 | Finger | |
| 4 | Finger tip | |
| 7 | Gear | Press-fit into the servo horn |
| 1 | Franka mount | Optional, only if mounting on a Franka arm. See [Notes](#notes) |

### Wires

| Qty | Wire | Length | Source | Notes |
|----:|------|--------|--------|-------|
| 1 | JST 1.25mm 3-pin cable | 150mm | Included with servos | Connects the two servos in the auxiliary gripper |
| 2 | Custom JST 1.25mm 3-pin Y-splice cable | 50mm trunk, 2x 200mm branches | Made from 3 pre-made wires + heat shrink each | One per gripper level, see Assembly step 2 |

## Print Settings (for all 3DP parts)

Speeds, temperatures and cooling vary by printer, so use your filament's stock profile for those.

| Setting | Value |
|---------|-------|
| Material | Bambu PLA Basic |
| Infill | 25%, cubic |
| Walls | 4 |
| Supports | Tree |
| Support material | Bambu Support for PLA (used for the support interface only) |
| Support interface offset (top Z distance) | 0 mm, for a smooth surface and tight tolerances |
| Layer height | 0.2 mm |
| Nozzle | 0.4 mm |

**Tips**

- Support for PLA only works as an interface layer. Print the rest of the support in PLA Basic (dual nozzle or AMS will be very helpful here). 
- Support for PLA is NOT OPTIONAL. Must use for smooth sliding. 
- The .step files are toleranced for our Bambulab H2C and H2D printer. If further adjustment is needed beyond the provided profiles, the tolerance is all parametrized in the Fusion360 file, feel free to adjust there. In the parametrization, there are three tolerance values used:
  - **Slide tolerance**: for all sliding dovetail joints to provide tight fit and alignment for sliding.
  - **Fit tolerance**: for fitting servo casing.
  - **Clear tolerance**: for anything that may slide past each other, but do not want to introduce further friction or alignment.
- The 3DP gears are only PLA! They will survive normal usage with torque exerted only from the servos. Do not try to manually open or close the parallel grippers too fast, otherwise you risk stripping or shearing the gear teeth.


## Assembly

### 1. Set up the 3D prints

Use the [Print Settings](#print-settings-for-all-3dp-parts) above.

**Print orientation**

| Part | Orientation |
|------|-------------|
| Base gripper rail | <img src="media/base_gripper_print_orientation.png" alt="Base gripper rail print orientation" width="462"> |
| Auxiliary gripper rail | <img src="media/aux_gripper_print_orientation.png" alt="Auxiliary gripper rail print orientation" width="462"> |
| Finger | <img src="media/finger_print_orientation.png" alt="Finger print orientation" width="462"> |
| Finger tip | <img src="media/finger_tip_print_orientation.png" alt="Finger tip print orientation" width="462"> |
| Gear | <img src="media/gear_print_orientation.png" alt="Gear print orientation" width="462"> |

### 2. Make the wires

The servo package includes one 150mm micro JST 1.25mm 3-pin cable per servo. You need to make two custom Y-splices (50mm trunk to 2x 200mm branches, micro JST 1.25mm 3-pin) from the pack of single-ended 200mm pre-made wires.

<img src="media/y_splice_diagram.svg" alt="Y-splice wire diagram" width="595">

The video covers this step from [8:24](https://youtu.be/W9SX5vXJZbE?t=504) to [12:43](https://youtu.be/W9SX5vXJZbE?t=763).

For each Y-splice (make two, one per gripper level):

1. Take 1 pre-made wire, cut it to 50mm and strip the last 10mm.
2. Take 2 more pre-made wires and strip only the last 10mm (do not cut them).
3. Slide a piece of heat shrink of the right size and length onto each of the three strands of the 50mm wire.
4. Twist together the matching colors of all three wires (the 50mm trunk and the two 200mm branches).
5. Apply minimal solder to keep the wire as flexible as possible.
6. Slide the heat shrink over each joint and apply heat to shrink it.
7. Check each finished splice with a multimeter in continuity mode: each pin of the trunk connector must beep to the same pin of both branch connectors, and no two different pins may beep to each other. A swapped or shorted wire can damage the servos.

### 3. Set up and label the servos

Each servo needs a unique ID before assembly, matching the DoF labels in the wiring diagram below. IDs are set with the `ft_servo_tools` CLI in [`scripts/`](https://github.com/generalroboticslab/Cartesian_Hand/tree/main/scripts/ft_servo_tools) of the [Cartesian_Hand](https://github.com/generalroboticslab/Cartesian_Hand) repo.

1. Connect **one servo at a time** to the driver board. `set-id` scans the bus to find its target, so more than one servo makes the choice ambiguous.
2. Run `python scripts/ft_servo_tools/cli.py set-id <port> <id>`, where `<port>` is your serial port (e.g. `/dev/ttyACM0`) and `<id>` is the servo's target position: 0-6, per the wiring tree diagram below. The servo jogs briefly to confirm the new ID took, then writes it to EEPROM (persists across power cycles).
3. Label the servo body with its assigned ID (e.g. tape + marker), so it's easy to identify during assembly.

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Joint | base jaw | base left finger | base right finger | z stage | aux jaw | aux left finger | aux right finger |

The auxiliary gripper rides the z stage. The software calls these seven joints DOF 0-6; [`docs/hardware.md`](../docs/hardware.md#configuring-a-new-hand) has the axis for each.

Repeat for all 7 servos before moving on. To confirm all 7 IDs are present, connect the full bus and run `python scripts/ft_servo_tools/cli.py scan <port>` (or use `... cli.py gui <port>` for a browser-based view of each servo's position, voltage and temperature).

### 4. Assemble the Cartesian Hand

Watch the assembly video alongside the steps below:

[![Cartesian Hand assembly video](https://img.youtube.com/vi/W9SX5vXJZbE/hqdefault.jpg)](https://youtu.be/W9SX5vXJZbE)

**Overall wiring structure**

<img src="media/wiring_tree_diagram.svg" alt="Servo board wiring tree" width="610">

1. Carefully remove all supports from the main body, using pliers, wire snippers and small pointy tools.
2. Prepare the servo horns (using the IDs set in step 3 above). Repeat for all 7 servos:
   1. Remove the servo horn and the support shaft from the servo. Discard the support shaft.
   2. Press-fit a gear into the screw holes of the servo horn.
   3. Place the servo horn back on the servo and screw it down tightly with an M2.5x10 screw.
3. Mount each servo onto its corresponding printed piece, as shown in the diagram _(coming soon)_:
   - Each servo in the base gripper rail and the auxiliary gripper rail uses 4 M2x6 screws.
   - Each servo in a finger uses 2 M2x6 screws, on a diagonal.
4. Connect the two servos in the auxiliary gripper with the 150mm pre-made cable, and tape it down snugly.
5. Assemble all the pieces.
6. Thread the Y-splice wire from the center servo on each rail to the fingers, as shown in the wiring structure diagram above.

The hand is now mechanically complete.

### 5. Verify and zero

1. Connect the full servo bus, power it, and run `python scripts/ft_servo_tools/cli.py scan <port>`. All 7 IDs (0-6) should answer.
2. Install the software (see the [main README](../README.md#installation)) and add an entry for your hand to `cartesian_hand/config.py`, following [Configuring a new hand](../docs/hardware.md#configuring-a-new-hand). Always pass `--hand my_hand`. Without it, a build with IDs 0-6 is silently treated as our lab's `hand_3`.
3. Run `python -m cartesian_hand.studio --hand my_hand`, open http://localhost:8081, and drag each slider a few mm. Each real joint should move the same way as the 3D model. Keep the moves small. Before zeroing, 0 mm is wherever the hand started, and the far end of each rail has no hard stop, so a carriage can run off its rail.
4. Zero the hand: `python -m cartesian_hand.studio --hand my_hand --task zero`. Zero again after every power cycle.
5. Optional: the default travel and torque floor are starting points. To tune them for your hand, see [Configuring a new hand](../docs/hardware.md#configuring-a-new-hand).

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| `scan` finds fewer than 7 servos | Check the Y-splice with a multimeter, and that every JST connector is fully seated. Re-run `scan` with servos connected one at a time to find the silent one. |
| Two servos answer to the same ID | A servo was not renamed. Disconnect all but one, and run `set-id` again. |
| A joint moves the wrong way in the studio | It is mounted differently from ours. See [DOF indexing](../docs/hardware.md#dof-indexing-and-orientation) before changing anything. |
| A finger or jaw binds or scrapes along its rail | Support interface not removed completely, or the print tolerance is too tight for your printer. Clean the dovetail, lightly sand the rails (see the [video](https://youtu.be/W9SX5vXJZbE?t=345)), or raise the slide tolerance in the Fusion360 file and reprint. |
| Gear teeth skip or strip | Gears are PLA only. Never move the grippers by hand quickly. Reprint the gear and press-fit it again. |
| Zeroing aborts | A joint never stalled against its hard stop. Nothing is written, and the previous calibration stays. Check for a jammed carriage, then see [Zeroing](../docs/hardware.md#zeroing). |

## Files

- `CAD/`: step files for each part (`base gripper rail`, `auxiliary gripper rail`, `finger`, `finger tip`, `gear`, `franka_mount`) and the Fusion360 design files (`cartesian_hand_v14_latest.f3z`, `franka_mount.f3d`)
- `media/`: media supporting assembly instructions

## Notes

- The mounting is designed for the Duke Humanoid V2. Feel free to change it in the CAD or adapt your own. See `franka_mount` in the CAD folder for reference. It's 3D-printed as well, and the hand is mounted with M3 screws that will self-tap into the printed franka_mount.
