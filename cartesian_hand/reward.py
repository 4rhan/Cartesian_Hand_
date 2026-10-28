"""Reward and episode metrics for `env.CapEnv`.

    python -m cartesian_hand.reward --obj water_500ml   # the teacher's reward terms

One `Reward` per env, reset with each episode, called once per tick. It reads
the answer sheet (cap turns, lift, the bottle's measurements) and the scene,
which is fine for the same reason an RL critic may: reward exists only in
training and evaluation, never on the deployed policy's input.

Shape of the reward
-------------------
**Progress is potential-based.** Turning and lifting pay the change in

    phi = turn * min(turns / turns_to_open, 1)  +  lift * min(lift / 10 mm, 1)

per tick, not its level. Shaping of that form leaves the optimal policy
unchanged (Ng, Harada & Russell 1999), so it guides exploration without
inventing a new objective -- which matters for residual RL, where the base
policy already solves much of the task and a reward that pays for something
else would pull the residual away from it. It also pays nothing for dithering:
turning back loses exactly what turning forward earned.

**Events pay once.** `release` on the tick the thread lets go, `success` on the
tick the cap is lifted clear. **Failures end the episode** with a penalty: the
cap `dropped` (touches the table, or falls below the neck's top beside the
bottle) or the bottle `tipped` past `TIP_DEG`. Ending there keeps a failed
episode from teaching anything after the failure.

**Jerk** is the third difference of each joint's commanded setpoint, in
mm/tick^3, squared and summed: a policy that flips its goals every tick pays
for it, one that ramps smoothly barely does. Smooth commands are what the
servos can follow and what a VLA's action chunks look like on the bench.

Episode metrics
---------------
`Metrics` accumulates over an episode and lands in `info["metrics"]` on its
last step: success and time to it, how far the cap got, the failure flags, the
return, and jerk, so an evaluation reports *why* an episode failed and not
only that it did.
"""
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .objects import ThreadMonitor

TIP_DEG = 20.0
"""Bottle axis this far from the table's normal counts as knocked over. Far
past the 3 degrees `episodes.HARD` stands a bottle at."""


@dataclass(frozen=True)
class RewardConfig:
    """Weights. Progress terms are per unit of progress, events per event."""

    turn: float = 1.0
    """For turning from closed to `turns_to_open`."""
    release: float = 1.0
    """Once, when the thread lets go."""
    lift: float = 1.0
    """For lifting the released cap `ThreadMonitor.OPENED_LIFT_MM`."""
    success: float = 5.0
    """Once, on the tick the cap is lifted clear."""
    drop: float = 5.0
    """Penalty, once, ending the episode."""
    tip: float = 5.0
    """Penalty, once, ending the episode."""
    jerk: float = 1e-3
    """Per mm^2/tick^6 of summed squared setpoint jerk. Measured: the scripted
    teacher's ramps sum to ~185 over a successful episode, so ~0.2 here; a
    policy flipping its goals at full speed every tick sums ~1300 *a tick*."""


@dataclass
class Metrics:
    """One episode's outcome. Plain floats and bools, so it logs and averages
    without unpacking."""

    success: bool = False
    time_to_success_s: float = float("nan")
    released: bool = False
    turn_fraction: float = 0.0
    """Highest `turns / turns_to_open` reached, capped at 1."""
    max_lift_mm: float = 0.0
    dropped: bool = False
    tipped: bool = False
    episode_return: float = 0.0
    jerk_rms: float = 0.0
    """RMS over ticks of the summed squared setpoint jerk, sqrt'd: mm/tick^3."""
    ticks: int = 0
    _jerk_sq: float = field(default=0.0, repr=False)

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


class Reward:
    """Per-tick reward and the episode's `Metrics`, for one env."""

    def __init__(self, cfg: RewardConfig = RewardConfig()):
        self.cfg = cfg

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData,
              monitor: ThreadMonitor, setpoint: np.ndarray) -> None:
        self.model, self.data, self.monitor = model, data, monitor
        self.bottle = monitor.bottle
        self._cap_geom = model.geom("cap").id
        self._floor_geom = model.geom("floor").id
        self._bottle_body = model.body("bottle").id
        self._history = [setpoint.copy()] * 3
        self.metrics = Metrics()
        self._phi = self._potential()
        self._was_released = False

    def __call__(self, setpoint: np.ndarray, hz: float
                 ) -> tuple[float, dict[str, float], bool]:
        """(reward, its terms, whether a failure ends the episode) for the tick
        just stepped, given the setpoint it was stepped toward."""
        c, m = self.cfg, self.metrics
        terms = {}

        phi = self._potential()
        terms["progress"] = phi - self._phi
        self._phi = phi

        released = self.monitor.released_at_s is not None
        terms["release"] = c.release if released and not self._was_released else 0.0
        self._was_released = released

        opened = self.monitor.opened
        terms["success"] = c.success if opened and not m.success else 0.0

        dropped = released and not opened and self._dropped()
        tipped = self._tipped()
        terms["drop"] = -c.drop if dropped else 0.0
        terms["tip"] = -c.tip if tipped else 0.0

        a, b, d = self._history
        jerk = setpoint - 3 * d + 3 * b - a
        jerk_sq = float(jerk @ jerk)
        terms["jerk"] = -c.jerk * jerk_sq
        self._history = [b, d, setpoint.copy()]

        reward = float(sum(terms.values()))
        m.ticks += 1
        m.episode_return += reward
        m._jerk_sq += jerk_sq
        m.jerk_rms = float(np.sqrt(m._jerk_sq / m.ticks))
        m.released = released
        m.turn_fraction = max(m.turn_fraction, self._turn_fraction())
        m.max_lift_mm = max(m.max_lift_mm, self.monitor.max_lift_mm)
        m.dropped |= dropped
        m.tipped |= tipped
        if opened and not m.success:
            m.success = True
            m.time_to_success_s = m.ticks / hz
        return reward, terms, dropped or tipped

    # -- what the scene says -------------------------------------------------

    def _turn_fraction(self) -> float:
        if self.monitor.released_at_s is not None:
            return 1.0       # released, the hinge is free and its angle moot
        turns = self.data.qpos[self.monitor._turn] / (2 * np.pi)
        return float(np.clip(turns / self.bottle.turns_to_open, 0.0, 1.0))

    def _potential(self) -> float:
        lift = (self.monitor._lift_mm(self.data)
                if self.monitor.released_at_s is not None else 0.0)
        return (self.cfg.turn * self._turn_fraction()
                + self.cfg.lift * float(np.clip(
                    lift / ThreadMonitor.OPENED_LIFT_MM, 0.0, 1.0)))

    def _dropped(self) -> bool:
        """The released cap is on the table, or has fallen off the neck's top
        to beside the bottle."""
        for i in range(self.data.ncon):
            pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if pair == {self._cap_geom, self._floor_geom}:
                return True
        return self.monitor._lift_mm(self.data) < -self.bottle.cap_height

    def _tipped(self) -> bool:
        axis = self.data.xmat[self._bottle_body].reshape(3, 3)[:, 2]
        normal = self.data.geom_xmat[self._floor_geom].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(axis @ normal, -1, 1)))) > TIP_DEG


def main(obj: str = "water_500ml", split: str = "train",
         index: int | None = None) -> None:
    """Run the scripted teacher once and print its reward terms and metrics.

    Args:
        obj: a catalog bottle, at the nominal pose. Ignored with --index.
        split: with --index, which split the episode comes from.
        index: run episode `index` of `split` instead.
    """
    import contextlib
    import io

    from . import env as E, objects
    e = E.CapEnv(split, cameras=False)
    options = ({"index": index} if index is not None else
               {"bottle": objects.load(obj)})
    totals: dict[str, float] = {}
    obs, info = e.reset(options=options)
    teacher = E.Teacher(e.cfg, info["privileged"])
    with contextlib.redirect_stdout(io.StringIO()):
        while (action := teacher(obs)) is not None:
            obs, _, terminated, truncated, info = e.step(action)
            for k, v in info["reward_terms"].items():
                totals[k] = totals.get(k, 0.0) + v
            if terminated or truncated:
                break
    print("[reward] summed terms: " + ", ".join(
        f"{k} {v:+.3f}" for k, v in totals.items()))
    print("[reward] metrics: " + ", ".join(
        f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
        for k, v in e.reward.metrics.as_dict().items()))


if __name__ == "__main__":
    import tyro
    tyro.cli(main, prog="python -m cartesian_hand.reward")
