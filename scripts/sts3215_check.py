"""Bench check for one Feetech STS3215: is it there, and does it turn.

The rest of the repo drives HLS-series servos through the compiled
`ft_servo` extension, whose register map is HLS (torque rides in the goal
packet, multi-turn positions). The STS3215 speaks the same packet format with
a different map, so this script talks to it directly over pyserial and needs
no build.

    uv pip install pyserial --python .venv/bin/python
    python scripts/sts3215_check.py                      # the only servo on the bus
    python scripts/sts3215_check.py --port /dev/ttyACM0 --id 3
    python scripts/sts3215_check.py --scan               # list IDs, move nothing

**It moves the servo**: a quarter turn (1024 counts) out and back by default.
Run it on a bare servo, or pass a smaller `--counts` if the horn is attached
to something.
"""

import argparse
import glob
import sys
import time

import serial

# STS (SMS_STS) register map, from the Feetech memory table.
REG_ID = 5
REG_MIN_ANGLE = 9       # EPROM, then max (11). Position goals outside are refused;
                        # both 0 is multi-turn, no limit
REG_OFFSET = 31         # EPROM. present = raw - offset, sign-magnitude on bit 11
REG_MODE = 33           # 0 position, 1 wheel (constant speed)
REG_TORQUE_ENABLE = 40
REG_ACC = 41            # then goal position (42), goal time (44), goal speed (46)
REG_PRESENT_POS = 56    # then speed (58), load (60), voltage (62), temperature (63)
REG_TORQUE_LIMIT = 48  # RAM, 0-1000; back to 1000 at power-up
REG_LOCK = 55           # 0 lets writes below 40 persist in EPROM, 1 locks them
REG_MOVING = 66

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03

COUNTS_PER_REV = 4096   # 0-4095 is one turn in position mode
MIN_VOLTS = 4.5         # STS3215 runs at 6-7.4V (or 12V for the C044 variant)
MAX_TEMP_C = 65

# Where zeroing puts the closed-end hard stop, in counts. Near the top of the
# turn so the 4095 -> 0 wrap sits just past the stop, outside the travel: a
# position-mode move never crosses it, so it can never go the long way round
# into the stop. 95 counts (~1.2mm) of margin for the carriage pressing into
# the stop. `stop_counts(-1)` is the hand's convention, mm rising as counts fall.
STOP_MARGIN = 95


def stop_counts(sign: int) -> int:
    return COUNTS_PER_REV - 1 - STOP_MARGIN if sign < 0 else STOP_MARGIN


def _sign_magnitude(raw: int, sign_bit: int = 15) -> int:
    """Position, speed and load set bit 15 for negative; not two's complement."""
    return -(raw & ~(1 << sign_bit)) if raw & (1 << sign_bit) else raw


def _to_sign_magnitude(value: int, sign_bit: int = 15) -> int:
    return (-value) | (1 << sign_bit) if value < 0 else value


def unwrap(prev: int, now: int) -> int:
    """Signed counts moved from `prev` to `now`, taking the short way round."""
    return (now - prev + COUNTS_PER_REV // 2) % COUNTS_PER_REV - COUNTS_PER_REV // 2


class Sts:
    """One serial port and the STS servos on it. Reads return None on no reply."""

    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        self.ser.close()

    def _send(self, sid: int, inst: int, params: bytes = b"") -> None:
        body = bytes([sid, len(params) + 2, inst]) + params
        self.ser.reset_input_buffer()
        self.ser.write(b"\xff\xff" + body + bytes([~sum(body) & 0xFF]))

    def _reply(self, sid: int) -> bytes | None:
        """The params of one status packet from `sid`, or None on timeout/corruption."""
        # Resync on the header: some adapters echo our own packet back first.
        while True:
            b = self.ser.read(1)
            if not b:
                return None
            if b != b"\xff" or self.ser.read(1) != b"\xff":
                continue
            head = self.ser.read(2)                     # id, length
            if len(head) < 2:
                return None
            rid, length = head
            rest = self.ser.read(length)                # error, params..., checksum
            if len(rest) < length or length < 2:
                return None
            if (~(rid + length + sum(rest[:-1])) & 0xFF) != rest[-1]:
                return None
            if rid != sid:
                continue                                # our echo, or another servo
            return rest[1:-1]

    def ping(self, sid: int) -> bool:
        self._send(sid, INST_PING)
        return self._reply(sid) is not None

    def read(self, sid: int, reg: int, n: int) -> bytes | None:
        self._send(sid, INST_READ, bytes([reg, n]))
        data = self._reply(sid)
        return data if data is not None and len(data) == n else None

    def write(self, sid: int, reg: int, data: bytes) -> bool:
        self._send(sid, INST_WRITE, bytes([reg]) + data)
        return self._reply(sid) is not None

    def scan(self, start: int = 0, end: int = 20) -> list[int]:
        return [i for i in range(start, end + 1) if self.ping(i)]

    def state(self, sid: int) -> dict | None:
        """Position, speed, load, volts, temperature in one read."""
        d = self.read(sid, REG_PRESENT_POS, 8)
        if d is None:
            return None
        return {
            "pos": _sign_magnitude(d[0] | d[1] << 8),
            "speed": _sign_magnitude(d[2] | d[3] << 8),
            "load": _sign_magnitude(d[4] | d[5] << 8, sign_bit=10),
            "volts": d[6] / 10,
            "temp": d[7],
        }

    def position(self, sid: int) -> int | None:
        s = self.state(sid)
        return None if s is None else s["pos"]

    def moving(self, sid: int) -> bool | None:
        d = self.read(sid, REG_MOVING, 1)
        return None if d is None else bool(d[0])

    def torque(self, sid: int, on: bool) -> bool:
        return self.write(sid, REG_TORQUE_ENABLE, bytes([int(on)]))

    def write16(self, sid: int, reg: int, value: int) -> bool:
        return self.write(sid, reg, bytes([value & 0xFF, value >> 8 & 0xFF]))

    def mode(self, sid: int, mode: int) -> bool:
        """With the lock on (the default) this does not persist across power."""
        return self.write(sid, REG_MODE, bytes([mode]))

    def get_mode(self, sid: int) -> int | None:
        d = self.read(sid, REG_MODE, 1)
        return None if d is None else d[0]

    def angle_limits(self, sid: int) -> tuple[int, int] | None:
        d = self.read(sid, REG_MIN_ANGLE, 4)
        return None if d is None else (d[0] | d[1] << 8, d[2] | d[3] << 8)

    def torque_limit(self, sid: int, limit: int) -> bool:
        return self.write16(sid, REG_TORQUE_LIMIT, limit)

    def wheel(self, sid: int, speed: int, acc: int = 50) -> bool:
        """Constant speed in counts/s, signed. Needs `mode(sid, 1)` first."""
        return self.write(sid, REG_ACC, bytes([acc])) and \
            self.write16(sid, 46, _to_sign_magnitude(speed))

    def offset(self, sid: int) -> int | None:
        d = self.read(sid, REG_OFFSET, 2)
        return None if d is None else _sign_magnitude(d[0] | d[1] << 8, sign_bit=11)

    def write_offset(self, sid: int, offset: int) -> bool:
        """EPROM: survives power cycles. Unlock, write, lock again."""
        if not -2047 <= offset <= 2047:
            raise ValueError(f"offset {offset} outside +-2047")
        ok = self.write(sid, REG_LOCK, b"\x00")
        ok = ok and self.write16(sid, REG_OFFSET, _to_sign_magnitude(offset, sign_bit=11))
        return self.write(sid, REG_LOCK, b"\x01") and ok

    def goto(self, sid: int, pos: int, speed: int, acc: int) -> bool:
        """Goal time 0 means speed governs the move, not a fixed duration."""
        pos = max(0, min(COUNTS_PER_REV - 1, pos))
        return self.write(sid, REG_ACC, bytes([
            acc, pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8,
        ]))


def wait_until_stopped(bus: Sts, sid: int, timeout: float = 3.0) -> int | None:
    time.sleep(0.1)                                     # let the move start
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and bus.moving(sid):
        time.sleep(0.02)
    return bus.position(sid)


def autodetect_port() -> str:
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if len(ports) != 1:
        sys.exit(f"found serial ports {ports or 'none'}; pass --port")
    return ports[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", help="default: the only /dev/ttyACM* or /dev/ttyUSB*")
    p.add_argument("--id", type=int, help="default: the only servo that answers a scan")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--counts", type=int, default=1024, help="how far to turn; 4096 = 1 rev")
    p.add_argument("--speed", type=int, default=1000, help="counts/s")
    p.add_argument("--acc", type=int, default=50, help="0-254, x100 counts/s^2")
    p.add_argument("--scan", action="store_true", help="list IDs 0-20 and exit")
    args = p.parse_args()

    port = args.port or autodetect_port()
    bus = Sts(port, args.baud)
    try:
        if args.scan:
            ids = bus.scan()
            print(f"{port} @ {args.baud}: found {ids}")
            for i in ids:
                print(f"  ID {i:3d}: {bus.state(i)}")
            return

        if args.id is None:
            ids = bus.scan()
            if len(ids) != 1:
                sys.exit(f"found servos {ids or 'none'} on {port} @ {args.baud}; pass --id")
            args.id = ids[0]

        s = bus.state(args.id)
        if s is None:
            sys.exit(f"no reply from ID {args.id} on {port} @ {args.baud}; try --scan")
        print(f"ID {args.id} on {port}: {s}")
        if s["volts"] < MIN_VOLTS:
            sys.exit(f"supply reads {s['volts']}V; is the servo power plugged in?")
        if s["temp"] > MAX_TEMP_C:
            sys.exit(f"servo is at {s['temp']}C; let it cool")

        # In wheel mode the goal position is ignored and the servo spins at the
        # goal speed instead, and reads come back without the offset. Found on
        # a servo that arrived in mode 1 (bench, 2026-10-20).
        mode = bus.get_mode(args.id)
        if mode != 0:
            print(f"ID {args.id} is in mode {mode}, not position mode; switching to 0")
            bus.torque(args.id, False)
            bus.mode(args.id, 0)
            time.sleep(0.4)                             # reads lag the switch
            s = bus.state(args.id)
            if s is None or bus.get_mode(args.id) != 0:
                sys.exit("could not switch to position mode")
            print(f"  now {s}")

        # The servo refuses goals outside its own angle limits, so turn within them.
        lo, hi = bus.angle_limits(args.id) or (0, 0)
        if lo == hi == 0:
            lo, hi = 0, COUNTS_PER_REV - 1
        start = s["pos"]
        if not lo - 5 <= start <= hi + 5:                # settling misses by a count or two
            # Possible after free spinning in wheel mode. Torque on will pull it to
            # the nearest limit, a short move, and the test runs from there.
            start = max(lo, min(hi, start))
            print(f"reads {s['pos']}, outside its angle limits {lo}-{hi}; "
                  f"starting from {start}")
        # Turn toward the middle of the limits so the goal never clips.
        step = args.counts if start < (lo + hi) // 2 else -args.counts
        goal = max(lo, min(hi, start + step))

        # Torque on snaps to whatever goal is in the register, so park it here first.
        bus.goto(args.id, start, args.speed, args.acc)
        bus.torque(args.id, True)
        try:
            if start != s["pos"]:
                print(f"pulled in to {wait_until_stopped(bus, args.id)}")
            print(f"turning {goal - start:+d} counts ({(goal - start) * 360 / COUNTS_PER_REV:+.0f} deg) to {goal}")
            bus.goto(args.id, goal, args.speed, args.acc)
            moved = wait_until_stopped(bus, args.id)
            print(f"now at {moved}")

            print(f"returning to {start}")
            bus.goto(args.id, start, args.speed, args.acc)
            print(f"back at {wait_until_stopped(bus, args.id)}")
        finally:
            bus.torque(args.id, False)
    finally:
        bus.close()

    if moved is None:
        print("FAIL: lost contact with the servo mid-move")
    elif abs(moved - goal) <= 20:
        print(f"PASS: turned {moved - start:+d} counts, {abs(moved - goal)} off target")
    else:
        print(f"FAIL: reached {moved}, expected {goal}")


if __name__ == "__main__":
    main()
