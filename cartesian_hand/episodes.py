"""Fixed train and test episodes: which bottle, standing where.

    python -m cartesian_hand.episodes                         # describe both splits
    python -m cartesian_hand.episodes --eval test --n 20      # teacher success on test
    python -m cartesian_hand.episodes --write-test --force    # regenerate the frozen file

An episode is one bottle variant plus one `objects.Placement`. Both splits draw
from the same wide ranges, `HARD`, and differ in where their randomness comes
from:

* **train** is drawn on demand from `random.Random("train:<index>")`. Any index
  is valid, so a data collector can keep asking for more.
* **test** is drawn once, from `random.Random("test:<index>")`, and frozen in
  `assets/objects/splits/test.json`. Frozen rather than regenerated because a
  change to `vary` or to `HARD` would otherwise silently swap the test bottles
  out from under every number already reported against them.

String seeds keep the two streams apart: Python seeds a `Random` from a
string through SHA-512, so "train:3" and "test:3" share nothing.

The ranges are wider than `objects.vary`'s defaults on purpose. A policy that
only ever sees bottles within 20% of one measurement has not been tested on
the next bottle someone brings to the bench.
"""
import dataclasses
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import tyro

from . import objects

BASES = ("water_500ml", "pill_bottle")
"""Catalog entries every episode is a variant of. Each episode picks one."""

HARD = dict(spread=0.35, log_factor=3.0, offset_mm=5.0, tilt_deg=3.0)
"""Sizes, masses, pitch, turns and friction +/-35% of the measurement;
resisting torque over a factor of 3 either way; +/-5 mm off centre; up to 3
degrees of tilt. Against `objects.vary`'s defaults of 20% and a factor of 2,
and `objects.PLACEMENT_RANGE`'s 3 mm and 2 degrees."""

TEST_SIZE = 50
TEST_FILE = os.path.join(objects.ROOT, "assets", "objects", "splits", "test.json")
SPLITS = ("train", "test")


@dataclass(frozen=True)
class Episode:
    split: str
    index: int
    bottle: objects.Bottle
    placement: objects.Placement

    @property
    def visual_seed(self) -> str:
        """Lighting and colours for this episode. Derived, not stored, so the
        frozen test file did not have to change to gain it."""
        return f"{self.split}:{self.index}:visual"

    def to_json(self) -> dict:
        return {"split": self.split, "index": self.index,
                "bottle": dataclasses.asdict(self.bottle),
                "placement": dataclasses.asdict(self.placement)}

    @staticmethod
    def from_json(raw: dict) -> "Episode":
        bottle = {k: tuple(v) if isinstance(v, list) else v
                  for k, v in raw["bottle"].items()}
        return Episode(raw["split"], raw["index"], objects.Bottle(**bottle),
                       objects.Placement(**raw["placement"]))


def draw(split: str, index: int) -> Episode:
    """Episode `index` of `split`, drawn fresh. Test episodes come from
    `test_set`, which reads the frozen file; this is how that file was made."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rng = random.Random(f"{split}:{index}")
    base = objects.load(rng.choice(BASES))
    bottle = objects.vary(base, rng, HARD["spread"],
                          name=f"{base.name}_{split}_{index:03d}",
                          log_factor=HARD["log_factor"])
    place = objects.sample_placement(rng, HARD["offset_mm"], HARD["tilt_deg"])
    return Episode(split, index, bottle, place)


def train(index: int) -> Episode:
    return draw("train", index)


def test_set(path: str = TEST_FILE) -> list[Episode]:
    """The frozen held-out episodes."""
    with open(path) as f:
        return [Episode.from_json(e) for e in json.load(f)["episodes"]]


def write_test_set(path: str = TEST_FILE, n: int = TEST_SIZE,
                   force: bool = False) -> None:
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; regenerating it changes the test "
                              f"set. Pass --force if that is the intent.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"_comment": "Held-out test episodes, frozen. Written by "
                               "`python -m cartesian_hand.episodes --write-test`; "
                               "do not edit by hand.",
                   "ranges": HARD, "bases": BASES,
                   "episodes": [draw("test", i).to_json() for i in range(n)]},
                  f, indent=1)
    print(f"wrote {n} test episodes to {path}")


def episodes(split: str, n: int) -> list[Episode]:
    return test_set()[:n] if split == "test" else [train(i) for i in range(n)]


def _teacher_opened(ep: Episode) -> tuple[Episode, bool, str]:
    import contextlib
    import io
    from . import env
    e = env.CapEnv(ep.split, cameras=False)
    with contextlib.redirect_stdout(io.StringIO()):
        info = env.run_episode(e, options={"index": ep.index})
    return ep, info["is_success"], e.monitor.summary()


def evaluate_teacher(split: str, n: int, workers: int) -> float:
    """Teacher success over the first `n` episodes of `split`: the ceiling a
    policy imitating it can be expected to reach."""
    wins = 0
    with ProcessPoolExecutor(workers) as pool:
        for ep, opened, summary in pool.map(_teacher_opened, episodes(split, n)):
            wins += opened
            b, p = ep.bottle, ep.placement
            print(f"  {b.name:24s} cap r{b.cap_radius:4.1f} turns "
                  f"{b.turns_to_open:4.2f} torque {b.break_torque:.3f} "
                  f"mu {b.friction:.2f}  off ({p.dx_mm:+4.1f},{p.dy_mm:+4.1f}) "
                  f"tilt {p.tilt_deg:3.1f}  {'OPENED' if opened else 'failed'}")
            if not opened:
                print(f"      {summary}")
    rate = wins / n
    print(f"[episodes] teacher on {split}: {wins}/{n} opened ({rate:.0%})")
    return rate


def main(eval: str | None = None, n: int = 20, workers: int = os.cpu_count() or 1,
         write_test: bool = False, force: bool = False) -> None:
    """Describe the splits, run the teacher on one, or regenerate the test file.

    Args:
        eval: `train` or `test`: run the scripted teacher on the first `n`
            episodes and report how many it opened.
        n: how many episodes `--eval` runs.
        workers: parallel processes for `--eval`.
        write_test: write the frozen test file. Refuses to overwrite without
            `--force`.
        force: allow `--write-test` to replace the existing test set.
    """
    if write_test:
        write_test_set(force=force)
        return
    if eval is not None:
        evaluate_teacher(eval, n, workers)
        return
    test = test_set()
    print(f"bases {BASES}, ranges {HARD}")
    print(f"test: {len(test)} frozen episodes in {TEST_FILE}")
    print("train: drawn on demand, any index; first three:")
    for i in range(3):
        ep = train(i)
        print(f"  {ep.bottle.name}: cap r{ep.bottle.cap_radius:.1f}, turns "
              f"{ep.bottle.turns_to_open:.2f}, placement off "
              f"({ep.placement.dx_mm:+.1f}, {ep.placement.dy_mm:+.1f}) mm")


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.episodes")
