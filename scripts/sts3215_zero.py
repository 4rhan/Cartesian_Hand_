"""Zero one STS3215: find its DOF's closed-end hard stop and make that 0mm.

The same protocol as `tasks/zero.py`, for one STS servo instead of the HLS
bus: creep into the stop at the DOF's torque floor, call it a stop when the
carriage stops advancing, then back off to mid travel.

    python scripts/sts3215_zero.py              # ID 2, hand_3's DOF 2
    python scripts/sts3215_zero.py --no-save    # find the stop, write nothing

Run it first, after mounting the servo and before `sts3215_twin.py`.

**Where the zero lives.** Not in `zero_offsets.json`, in the servo: its
position-offset register is shifted until the stop reads `stop_counts()`, near
the top of the turn. Two reasons. The encoder is absolute within a turn, so a
zero kept in the servo is still right after a power cycle (the HLS hand's
saved offsets go stale by whole turns; this one cannot). And it puts the
4095 -> 0 wrap just past the stop, outside the travel, which is what makes
position mode safe: a goal on the far side of the wrap would otherwise be
reached the long way round, straight through the stop.

**The seek runs in wheel mode**, not position mode, for the same wrap reason:
before zeroing nobody knows where the wrap is. Progress is tracked by unwrapping
successive reads, and the servo is stopped when that progress stalls.

**It needs a hard stop.** A bare servo on the bench has none and spins until
the timeout, which is a correct FAIL. Hold the horn to fake one if you want to
watch it succeed: the torque cap is the DOF's floor, 25% on a finger.
"""

import argparse
import sys
import time

from cartesian_hand.config import LABELS, get_hand
from sts3215_check import (COUNTS_PER_REV, MAX_TEMP_C, MIN_VOLTS, Sts,
                           autodetect_port, stop_counts, unwrap)

TICK_S = 0.02
WARMUP_S = 0.6           # acceleration ramp before stall detection starts
STALL_WINDOW_S = 0.5
STALL_FRACTION = 0.25    # under this share of the expected travel = stuck
WRONG_WAY_COUNTS = 15    # moving away from the stop by this much aborts
MODE_SETTLE_S = 0.4      # reads stay in the old frame this long after a mode write


def seek(bus: Sts, sid: int, speed: int, timeout: float) -> bool:
    """Drive at `speed` (signed counts/s) until progress stalls. False on
    timeout, wrong way or lost contact. Leaves the servo in position mode.

    Wheel mode reports the RAW position, offset not applied (position mode
    subtracts it), so nothing read in here is in the frame of a position-mode
    read. The first read is taken after the switch, or the offset shows up
    as a jump on the first tick. Read the stop after this returns.
    """
    bus.mode(sid, 1)
    # The frame flips up to ~0.2s AFTER the mode write (bench, 2026-10-19:
    # 274 at once, 2085 at +0.2s, offset 1813). Back to position mode is
    # immediate, but `MODE_SETTLE_S` is waited there too.
    time.sleep(MODE_SETTLE_S)
    prev = bus.position(sid)
    if prev is None:
        return False
    bus.torque(sid, True)
    bus.wheel(sid, speed)
    travelled, history = 0, []
    t0 = time.monotonic()
    try:
        while (t := time.monotonic() - t0) < timeout:
            time.sleep(TICK_S)
            now = bus.position(sid)
            if now is None:
                print("lost contact")
                return False
            travelled += unwrap(prev, now)
            prev = now
            history.append((t, travelled))
            if travelled * speed < 0 and abs(travelled) > WRONG_WAY_COUNTS:
                print(f"moved {travelled:+d} counts, AWAY from the stop: the wheel "
                      f"direction is not what this script assumes")
                return False
            if t < WARMUP_S:
                continue
            past = [d for ts, d in history if ts <= t - STALL_WINDOW_S]
            if past and abs(travelled - past[-1]) < STALL_FRACTION * abs(speed) * STALL_WINDOW_S:
                print(f"stalled after {travelled:+d} counts in {t:.1f}s")
                return True
            if len(history) % 100 == 0:
                print(f"  {t:5.1f}s / {timeout:.0f}s  {travelled:+6d} counts, still "
                      f"moving -- no stop yet", flush=True)
        print(f"no stall in {timeout:.0f}s ({travelled:+d} counts): no hard stop?")
        return False
    finally:
        bus.wheel(sid, 0)
        time.sleep(0.1)
        bus.mode(sid, 0)
        time.sleep(MODE_SETTLE_S)


def set_stop(bus: Sts, sid: int, sign: int) -> bool:
    """Shift the offset so the present reading becomes `stop_counts(sign)`.

    present = raw - offset, so raising the offset lowers the reading. Checked
    against a read-back anyway, and undone if the reading did not land.
    """
    target = stop_counts(sign)
    old, here = bus.offset(sid), bus.position(sid)
    if old is None or here is None:
        return False
    new = (old + here - target) % COUNTS_PER_REV
    new = new - COUNTS_PER_REV if new > COUNTS_PER_REV // 2 else new
    if abs(new) > 2047:                                 # the one value +-2048
        new = new - 1 if new > 0 else new + 1
    bus.write_offset(sid, new)
    got = bus.position(sid)
    if got is not None and abs(unwrap(target, got)) <= 2:
        print(f"offset {old} -> {new}: the stop now reads {got} (target {target})")
        return True
    print(f"offset {new} read back {got}, not {target}; restoring {old}")
    bus.write_offset(sid, old)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", help="default: the only /dev/ttyACM* or /dev/ttyUSB*")
    p.add_argument("--id", type=int, default=2)
    p.add_argument("--hand", default=None, help="config whose ID block names the DOF")
    p.add_argument("--dof", type=int, help="default: id - first_servo_id")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--seek-speed", type=float, default=0.75,
                   help="mm/s into the stop; the force knob, see tasks/zero.py")
    p.add_argument("--torque", type=int, help="cap, 0-1000; default: the DOF's floor")
    p.add_argument("--park-speed", type=int, default=300, help="counts/s back out")
    p.add_argument("--timeout", type=float, help="seconds; default: 1.3x the rail at seek speed")
    p.add_argument("--no-save", action="store_true", help="find the stop, keep the offset")
    args = p.parse_args()

    cfg = get_hand(args.hand) if args.hand else get_hand()
    dof = args.id - cfg.first_servo_id if args.dof is None else args.dof
    if not 0 <= dof < cfg.n_dof:
        sys.exit(f"ID {args.id} is outside {cfg.name}'s block {cfg.servo_ids}; pass --dof")
    sign = int(cfg.orientations()[dof].item())
    torque = args.torque or int(cfg.gain_vector("torque_min_to_move")[dof].item())
    # The closed end is mm -> 0, which in counts is -sign.
    speed = round(-sign * args.seek_speed * cfg.counts_per_mm)
    timeout = args.timeout or cfg.travel_mm[dof] * 1.3 / args.seek_speed

    port = args.port or autodetect_port()
    bus = Sts(port, args.baud)
    try:
        s = bus.state(args.id)
        if s is None:
            sys.exit(f"no reply from ID {args.id} on {port}; run sts3215_check.py --scan")
        print(f"ID {args.id} -> {cfg.name} DOF {dof} ({LABELS[dof]}): {s}")
        if s["volts"] < MIN_VOLTS or s["temp"] > MAX_TEMP_C:
            sys.exit("supply too low or servo too hot; refusing to seek")

        print(f"seeking the stop at {args.seek_speed} mm/s ({speed:+d} counts/s), "
              f"torque cap {torque}, timeout {timeout:.0f}s")
        print("a bare servo has no stop: pinch the horn to make one, or it "
              "FAILs at the timeout")
        bus.torque_limit(args.id, torque)
        if not seek(bus, args.id, speed, timeout):
            print("FAIL: no hard stop found; the offset is unchanged")
            return
        stop = bus.position(args.id)                    # position mode again: offset applied
        if stop is None:
            print("FAIL: lost contact at the stop")
            return

        if args.no_save:
            print(f"stop reads {stop}; --no-save, offset unchanged")
            here = stop
        else:
            bus.torque(args.id, False)                  # the offset write moves the goal frame
            if not set_stop(bus, args.id, sign):
                print("FAIL: offset write did not take")
                return
            here = stop_counts(sign)

        # Park half a rail out, in position mode. Safe now: with the stop near the
        # top of the turn, mid travel is on the same side of the wrap.
        park = here + round(sign * cfg.upper()[dof].item() / 2 * cfg.counts_per_mm)
        if not 0 <= park < COUNTS_PER_REV:
            print(f"park goal {park} crosses the wrap; leaving it at the stop")
        else:
            bus.torque_limit(args.id, 1000)
            bus.goto(args.id, here, args.park_speed, 50)
            bus.torque(args.id, True)
            bus.goto(args.id, park, args.park_speed, 50)
            time.sleep(abs(park - here) / args.park_speed + 0.5)
            print(f"parked at {bus.position(args.id)} (goal {park}, "
                  f"{cfg.upper()[dof].item() / 2:.1f} mm)")
        print("PASS: zeroed" if not args.no_save else "PASS: stop found")
    finally:
        bus.torque(args.id, False)
        bus.torque_limit(args.id, 1000)
        bus.mode(args.id, 0)
        bus.close()


if __name__ == "__main__":
    main()
