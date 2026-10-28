"""One STS3215 and its joint in the MuJoCo model, side by side.

    python scripts/sts3215_twin.py              # drag the slider, the servo follows
    python scripts/sts3215_twin.py --teach      # torque off: turn the horn by hand
    python scripts/sts3215_twin.py --no-viewer  # print the mirror, no window

Two directions, the same ones `studio` runs for the whole HLS hand:

    ctrl   what you asked for   viewer Control slider -> mm -> counts -> servo
    qpos   what the servo did   servo -> counts -> mm -> the model's joint

The model's pose is never the slider: it is what came back off the bus, so a
joint that lags or stops short of the slider is the servo doing that, not the
render. Open the viewer's **Control** panel (right side) to find the slider.

Which joint: the hand config maps servo ID to DOF as `id - first_servo_id`, so
on hand_3 (IDs 0-6) servo 2 is DOF 2, the base finger. `--dof` overrides it.

**Run `sts3215_zero.py` first.** It shifts the servo's own offset so the
closed-end hard stop reads `stop_counts()` (4000 on this DOF), and this script
takes that reading as 0mm. The zero lives in the servo, so it survives power
cycles. Unzeroed, the millimetres are arbitrary but the mirror still works.

**One turn is the reach.** The STS3215 in position mode reads 0-4095, one
revolution, which on the 16mm pinion is 50.3mm of rack -- less than the
finger's 55mm -- and the stop margin takes ~1.2mm of that. The slider stops at
~49mm; the rest of the rail needs multi-turn mode.

**It moves the servo** unless `--teach`. Torque comes on holding the present
position, and goes off at exit.
"""

import argparse
import time

import mujoco
import mujoco.viewer

from cartesian_hand.config import LABELS, get_hand
from cartesian_hand.mjcf import MM_PER_M, mjcf_path, qpos_addrs
from sts3215_check import COUNTS_PER_REV, Sts, autodetect_port, stop_counts

CONTROL_HZ = 50
REPORT_HZ = 4
MAX_SILENT_STEPS = 25            # half a second at 50 Hz


class Mapping:
    """Counts <-> mm for one DOF, with `stop_counts` as the closed end."""

    def __init__(self, cfg, dof: int):
        self.cpm = cfg.counts_per_mm
        self.sign = cfg.orientations()[dof].item()
        # Counts run against +mm when sign is -1, so the closed end is high.
        self.zero = stop_counts(self.sign)
        reach = self.zero if self.sign < 0 else COUNTS_PER_REV - 1 - self.zero
        self.max_mm = min(cfg.travel_mm[dof], reach / self.cpm)

    def mm(self, counts: int) -> float:
        return (counts - self.zero) * self.sign / self.cpm

    def counts(self, mm: float) -> int:
        mm = max(0.0, min(self.max_mm, mm))
        return round(self.zero + mm * self.cpm * self.sign)


class _NoViewer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", help="default: the only /dev/ttyACM* or /dev/ttyUSB*")
    p.add_argument("--id", type=int, default=2)
    p.add_argument("--hand", default=None, help="config whose ID block names the DOF")
    p.add_argument("--dof", type=int, help="default: id - first_servo_id")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--speed", type=int, default=1000, help="counts/s")
    p.add_argument("--acc", type=int, default=50, help="0-254, x100 counts/s^2")
    p.add_argument("--teach", action="store_true", help="torque off, send no goals")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--xml", help="MJCF path; default is the bundled model")
    args = p.parse_args()

    cfg = get_hand(args.hand) if args.hand else get_hand()
    dof = args.id - cfg.first_servo_id if args.dof is None else args.dof
    if not 0 <= dof < cfg.n_dof:
        raise SystemExit(f"ID {args.id} is outside {cfg.name}'s block "
                         f"{cfg.servo_ids}; pass --dof")
    m = Mapping(cfg, dof)

    model = mujoco.MjModel.from_xml_path(mjcf_path(args.xml))
    data = mujoco.MjData(model)
    joints = qpos_addrs(model)[dof]
    # Slider range is what the servo can reach in one turn, so the slider
    # cannot ask for a goal the clamp in `Mapping.counts` would then refuse.
    model.actuator_ctrlrange[dof] = (0.0, m.max_mm / MM_PER_M)
    model.actuator_ctrllimited[dof] = 1

    port = args.port or autodetect_port()
    bus = Sts(port, args.baud)
    s = bus.state(args.id)
    if s is None:
        bus.close()
        raise SystemExit(f"no reply from ID {args.id} on {port}; "
                         f"run sts3215_check.py --scan")
    counts = s["pos"]
    print(f"ID {args.id} -> {cfg.name} DOF {dof} ({LABELS[dof]})"
          f" joint {joints}: {counts} counts = {m.mm(counts):.1f} mm, "
          f"{s['volts']}V {s['temp']}C")

    # Slider parked on the present pose, goal written before torque: enabling
    # torque snaps to whatever goal is in the register.
    data.ctrl[dof] = m.mm(counts) / MM_PER_M
    sent = counts
    if args.teach:
        bus.torque(args.id, False)
        print("teach: torque off, turn the horn by hand")
    else:
        bus.goto(args.id, counts, args.speed, args.acc)
        bus.torque(args.id, True)
        print("drag the slider in the viewer's Control panel")

    ctx = _NoViewer() if args.no_viewer else mujoco.viewer.launch_passive(model, data)
    silent, load = 0, 0
    next_report = time.monotonic()
    try:
        with ctx as view:
            while view.is_running():
                tick = time.monotonic()

                if not args.teach:
                    goal = m.counts(data.ctrl[dof] * MM_PER_M)
                    if goal != sent:
                        bus.goto(args.id, goal, args.speed, args.acc)
                        sent = goal

                s = bus.state(args.id)
                if s is None:
                    silent += 1
                    if silent >= MAX_SILENT_STEPS:
                        raise RuntimeError(f"ID {args.id} stopped answering on {port}")
                else:
                    silent, counts, load = 0, s["pos"], s["load"]

                for a in joints:
                    data.qpos[a] = m.mm(counts) / MM_PER_M
                mujoco.mj_forward(model, data)
                view.sync()

                if tick >= next_report:
                    want = "-" if args.teach else f"{m.mm(sent):5.1f}"
                    print(f"goal {want} mm   actual {m.mm(counts):5.1f} mm   "
                          f"counts {counts:4d}   load {load:+4d}", flush=True)
                    next_report = tick + 1 / REPORT_HZ

                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        bus.torque(args.id, False)
        bus.close()
        print("torque off")


if __name__ == "__main__":
    main()
