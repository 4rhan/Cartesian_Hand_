"""Driver for Feetech STS-series servos (STS3215), in pure Python over pyserial.

Same method surface as the compiled HLS driver (`ft_servo_ext.FtServo`) and
`servo.MockServo`, so everything above `servo.open_driver` runs unchanged:

    bus = open_driver("/dev/ttyACM0", sts=True)
    bus.read_all([1, 2, 3, 4])           # [(pos, speed, load) | None, ...]
    bus.set_positions(ids, pos, speed=300, acc=50, torque=500)

Same packet format as HLS, different register map. Two differences reach the
surface:

**Torque is a register, not part of the goal.** HLS takes a force cap with
every goal. The STS has none in its goal block, but its TORQUE_LIMIT register
(48) sits directly after it -- acc 41, position 42, time 44, speed 46, limit 48
-- so `set_positions` writes all five in one 9-byte sync-write slice, and the
`torque` argument still means what it means on HLS: a 0-1000 cap riding with
the goal.

**Positions are one turn, 0-4095,** in position mode, clamped by the servo to
its own angle limits (registers 9-12). The hand's rails need more than a turn
(55mm against 50.3mm per rev on a 16mm pinion); multi-turn is not handled here.

Sync-read (one request, every servo replies) and sync-write (one broadcast, no
reply) are used for the batch calls, as in the HLS driver. Everything is
behind one lock: the serial port is the bus.
"""

import threading
import time
from collections.abc import Sequence

import serial

# STS (SMS_STS) register map, from the Feetech memory table.
REG_MODE = 33              # 0 position, 1 wheel
REG_TORQUE_ENABLE = 40
REG_GOAL = 41              # acc, position(2), time(2), speed(2), torque limit(2)
GOAL_LEN = 9
REG_TORQUE_LIMIT = 48
REG_PRESENT = 56           # position(2), speed(2), load(2)
PRESENT_LEN = 6
REG_VOLTAGE = 62
REG_TEMPERATURE = 63

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST = 0xFE

Gain = int | Sequence[int]
Reading = tuple[int, int, int] | None


def _decode(raw: int, sign_bit: int) -> int:
    """Sign-magnitude, not two's complement: bit `sign_bit` set means negative."""
    return -(raw & ~(1 << sign_bit)) if raw & (1 << sign_bit) else raw


def _encode(value: int, sign_bit: int = 15) -> int:
    return (-value) | (1 << sign_bit) if value < 0 else value


def _word(value: int) -> list[int]:
    return [value & 0xFF, value >> 8 & 0xFF]


class StsServo:
    """One serial port and the STS servos on it. Reads return None, never a
    placeholder number, when a servo does not answer."""

    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.02):
        self.port = port
        self._timeout = timeout
        self._ser = serial.Serial(port, baud, timeout=timeout)
        self._lock = threading.Lock()

    # ── Packets ───────────────────────────────────────────────────────────────

    def _send(self, sid: int, inst: int, params: Sequence[int] = ()) -> None:
        body = bytes([sid, len(params) + 2, inst, *params])
        self._ser.reset_input_buffer()
        self._ser.write(b"\xff\xff" + body + bytes([~sum(body) & 0xFF]))

    def _packet(self, deadline: float) -> tuple[int, bytes] | None:
        """Next well-formed status packet as (id, params), or None by `deadline`."""
        while time.monotonic() < deadline:
            if self._ser.read(1) != b"\xff":
                continue
            if self._ser.read(1) != b"\xff":
                continue
            head = self._ser.read(2)
            if len(head) < 2 or head[1] < 2:
                continue
            sid, length = head
            rest = self._ser.read(length)                   # error, params, checksum
            if len(rest) < length:
                continue
            if (~(sid + length + sum(rest[:-1])) & 0xFF) != rest[-1]:
                continue
            return sid, rest[1:-1]
        return None

    def _reply(self, sid: int) -> bytes | None:
        deadline = time.monotonic() + self._timeout
        while (p := self._packet(deadline)) is not None:
            if p[0] == sid:                                 # skip echoes and strays
                return p[1]
        return None

    def _read(self, sid: int, reg: int, n: int) -> bytes | None:
        with self._lock:
            self._send(sid, INST_READ, (reg, n))
            data = self._reply(sid)
        return data if data is not None and len(data) == n else None

    def _write(self, sid: int, reg: int, data: Sequence[int]) -> bool:
        with self._lock:
            self._send(sid, INST_WRITE, (reg, *data))
            return self._reply(sid) is not None

    def _sync_write(self, ids: Sequence[int], reg: int,
                    slices: Sequence[Sequence[int]]) -> None:
        params = [reg, len(slices[0])]
        for sid, data in zip(ids, slices):
            params += [sid, *data]
        with self._lock:
            self._send(BROADCAST, INST_SYNC_WRITE, params)  # broadcast: no reply

    # ── Commands ──────────────────────────────────────────────────────────────

    def set_positions(self, ids: Sequence[int], pos: Sequence[int],
                      speed: Gain = 0, acc: Gain = 50, torque: Gain = 500) -> bool:
        """Goal, speed, acc and torque cap for every servo, one broadcast.

        Gains are one number for all, or one per servo, as on HLS -- and like
        the compiled driver's overloads, all sequences or all scalars, never a
        mix. `speed=0` is as fast as it can, not "do not move".
        """
        n = len(ids)
        if len(pos) != n:
            raise ValueError(f"ids and pos differ in length: {n} vs {len(pos)}")
        seq = [isinstance(g, (list, tuple)) for g in (speed, acc, torque)]
        if any(seq) and not all(seq):
            raise TypeError("set_positions gains must be all sequences or all scalars")

        def per_servo(gain: Gain, name: str) -> list[int]:
            if isinstance(gain, (list, tuple)):
                if len(gain) != n:
                    raise ValueError(f"{name} has {len(gain)} values, expected {n}")
                return [int(g) for g in gain]
            return [int(gain)] * n

        slices = [[a, *_word(_encode(int(p))), 0, 0, *_word(s), *_word(t)]
                  for p, s, a, t in zip(pos, per_servo(speed, "speed"),
                                        per_servo(acc, "acc"),
                                        per_servo(torque, "torque"))]
        if n:
            self._sync_write(ids, REG_GOAL, slices)
        return True

    def set_position(self, sid: int, pos: int, speed: int = 0, acc: int = 50,
                     torque: int = 500) -> bool:
        return self._write(sid, REG_GOAL,
                           [acc, *_word(_encode(int(pos))), 0, 0, *_word(speed), *_word(torque)])

    def enable_torque(self, sid: int, on: bool) -> bool:
        return self._write(sid, REG_TORQUE_ENABLE, [int(on)])

    def enable_torques(self, ids: Sequence[int], on: bool) -> bool:
        return all([self.enable_torque(sid, on) for sid in ids])

    def set_torque_limit(self, sid: int, limit: int) -> bool:
        """The same register `set_positions`'s torque writes; kept for parity."""
        return self._write(sid, REG_TORQUE_LIMIT, _word(int(limit)))

    def set_torque_limits(self, ids: Sequence[int], limits: Sequence[int]) -> None:
        if len(limits) != len(ids):
            raise ValueError(f"ids and limits differ in length: {len(ids)} vs {len(limits)}")
        for sid, limit in zip(ids, limits):
            self.set_torque_limit(sid, limit)

    def set_mode(self, sid: int, mode: int) -> bool:
        """0 position, 1 wheel. Reads lag the switch by up to ~0.2s."""
        return self._write(sid, REG_MODE, [mode])

    # ── Reads ─────────────────────────────────────────────────────────────────

    def read_all(self, ids: Sequence[int]) -> list[Reading]:
        """(position, speed, load) per servo in `ids` order, None for any that
        did not answer. One sync-read: one request, one reply per servo."""
        if not ids:
            return []
        got: dict[int, bytes] = {}
        with self._lock:
            self._send(BROADCAST, INST_SYNC_READ, (REG_PRESENT, PRESENT_LEN, *ids))
            # Each reply is 12 bytes, ~0.12ms at 1Mbaud, plus the servo's own
            # return delay; one timeout covers a silent servo once, not per servo.
            deadline = time.monotonic() + self._timeout + 0.0005 * len(ids)
            want = set(ids)
            while want and (p := self._packet(deadline)) is not None:
                sid, data = p
                if sid in want and len(data) == PRESENT_LEN:
                    got[sid] = data
                    want.discard(sid)
        out: list[Reading] = []
        for sid in ids:
            d = got.get(sid)
            out.append(None if d is None else (
                _decode(d[0] | d[1] << 8, 15),
                _decode(d[2] | d[3] << 8, 15),
                _decode(d[4] | d[5] << 8, 10)))
        return out

    def read_positions(self, ids: Sequence[int]) -> list[int | None]:
        return [None if r is None else r[0] for r in self.read_all(ids)]

    def read_position(self, sid: int) -> int | None:
        r = self.read_all([sid])[0]
        return None if r is None else r[0]

    def read_speed(self, sid: int) -> int | None:
        r = self.read_all([sid])[0]
        return None if r is None else r[1]

    def read_load(self, sid: int) -> int | None:
        r = self.read_all([sid])[0]
        return None if r is None else r[2]

    def get_voltage(self, sid: int) -> int | None:
        """Tenths of a volt (74 = 7.4V), or None."""
        d = self._read(sid, REG_VOLTAGE, 1)
        return None if d is None else d[0]

    def get_temperature(self, sid: int) -> int | None:
        d = self._read(sid, REG_TEMPERATURE, 1)
        return None if d is None else d[0]

    def ping(self, sid: int) -> int:
        with self._lock:
            self._send(sid, INST_PING)
            return sid if self._reply(sid) is not None else -1

    def scan(self, start_id: int = 0, end_id: int = 253) -> list[int]:
        return [i for i in range(start_id, end_id + 1) if self.ping(i) >= 0]

    def close(self) -> None:
        """Does NOT release torque, matching the HLS driver."""
        self._ser.close()
