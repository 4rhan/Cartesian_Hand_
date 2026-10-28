"""The cap-opening task as a gymnasium environment, for a VLA and for RL.

    python -m cartesian_hand.env --obj water_500ml          # one teacher episode
    python -m cartesian_hand.env --split test --index 3     # a test-split episode
    python -m cartesian_hand.env --obj water_500ml --check  # prove nothing leaks

    env = CapEnv(split="train")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

`sim.run` hands a controller the loop; this hands the loop to the caller, one
control tick per `step`, which is the shape a learned policy (pi0.5), a data
recorder and an RL library all need. The physics per tick is `sim`'s: the same
model loading, the same setpoint profiling, the same effort-as-force-cap.

Observation and action
----------------------
`obs` is everything the policy under test gets, and all of it exists on the
bench:

    state    float32 [7]            joint positions, mm, in DOF order
    images   {base, wrist}: uint8 [224, 224, 3]   (with cameras on)
    prompt   str                    the instruction

`action` is float32 [14]: seven position goals in mm, then seven effort limits
in 0-1 -- what the servo bus takes, `goal` and `torque / 1000`. The env walks
each joint's setpoint toward its goal at no more than the servo's top speed
(`action_speed_mm_s`, 84 mm/s), as the servo's own profiler would. A policy
that wants a slower move sends nearer goals; one that sends a goal 40 mm away
gets the same ramp the hardware would give it.

The answer sheet
----------------
`info["privileged"]` is the `Privileged` answer sheet: the bottle's
measurements, its placement, how far the cap has turned, where it is. Three
readers may have it -- the scripted teacher that generates demonstrations, the
scorer, and an RL critic (asymmetric actor-critic trains the critic on it and
deploys only the actor). The policy under test never does. A policy that saw
`turns_to_open` or `cap_offset` would succeed in sim for a reason that does
not exist on the bench.

That split is structural: `obs` has exactly `OBS_KEYS`, `check_obs` asserts
none of them is a privileged name, and the teacher reaches the truth only
through `objects.teacher_config` at construction.

Episodes
--------
`reset(seed=s)` on a split draws episode `s` of it: train episode `s`, or test
episode `s mod 50`. `reset()` with no seed draws an index from the env's own
`np_random`. `options` overrides either: `{"index": i}`, or `{"bottle": b,
"placement": p, "visual_seed": v}` for a bottle that is in no split. An
episode ends `terminated` once the cap is lifted (`info["is_success"]`) or
dropped, or the bottle tipped over; `truncated` at `max_seconds`. Reward and
the episode's metrics are `reward.py`'s: `info["reward_terms"]` every step,
`info["metrics"]` on the last.
"""
import os
import string
import time
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
import torch
import tyro
from gymnasium import spaces

from . import episodes, objects, sim, tasks
from .reward import Reward, RewardConfig
from .config import DEFAULT_HAND, HandConfig, get_hand
from .mjcf import MM_PER_M, qpos_addrs
from .policy import PolicyRunner
from .primitives import SERVO_NO_LOAD_RPM

PROMPT = "unscrew the cap and lift it off the bottle"

def action_speed_mm_s(hand: HandConfig) -> float:
    """How fast a joint's setpoint may walk toward its goal: the servo's rated
    no-load top speed, `primitives.SERVO_NO_LOAD_RPM` -- the same speed every
    free move in `Sequence` runs at, so the scripted teacher's own ramps never
    hit it and its demonstrations replay through this action space unchanged."""
    return SERVO_NO_LOAD_RPM / 60.0 * hand.counts_per_rev / hand.counts_per_mm

OBS_KEYS = ("state", "images", "prompt")


@dataclass(frozen=True)
class Privileged:
    """The answer sheet: the scripted teacher, the scorer and an RL critic."""

    bottle: objects.Bottle
    placement: objects.Placement
    cap_turns: float
    cap_lift_mm: float
    """Cap's lower edge above the neck's top, along the bottle's axis; 0 while
    still threaded."""
    released: bool
    opened: bool
    """The episode's score: released and lifted clear."""


FORBIDDEN = {f for cls in (objects.Bottle, objects.Placement, Privileged)
             for f in cls.__dataclass_fields__}
"""Names that must never appear as an observation key."""


def check_obs(obs: dict) -> None:
    """Raise if an observation carries anything but `OBS_KEYS`, or a key named
    like a privileged field."""
    keys = set(obs)
    if not keys <= set(OBS_KEYS) or keys & FORBIDDEN:
        raise AssertionError(f"observation keys {sorted(keys)}; allowed "
                             f"{OBS_KEYS}, never {sorted(keys & FORBIDDEN)}")
    images = obs.get("images", {})
    if set(images) & FORBIDDEN:
        raise AssertionError(f"image key leaks {sorted(set(images) & FORBIDDEN)}")


class CapEnv(gym.Env):
    """Open one bottle's cap per episode, stepped by the caller at control rate."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, split: str = "train", hand: str = DEFAULT_HAND,
                 cameras: bool = True, render_every: int = 1,
                 max_seconds: float = 30.0, visual_randomization: bool = True,
                 reward: RewardConfig = RewardConfig(),
                 render_mode: str | None = None):
        """
        Args:
            split: `train` or `test`, which `episodes` split `reset` draws from.
            hand: which entry in `config.HANDS` supplies travel and control rate.
            cameras: render `objects.CAMERAS` into `obs["images"]` at
                `objects.IMAGE_SIZE`. Two views cost ~13x a tick's physics (see
                docs/env.md), so a state-only RL run or the teacher turns it off.
            render_every: render on every Nth tick and repeat the last images in
                between. A policy that infers an action chunk at 10 Hz on this
                50 Hz loop needs `render_every=5`, not frames it never reads.
            max_seconds: simulated time before an episode is truncated.
            visual_randomization: redraw lighting and colours per episode from
                its `Episode.visual_seed`.
            reward: weights for `reward.Reward`'s terms.
            render_mode: `rgb_array` makes `render()` return both views side by
                side. Needs `cameras`.
        """
        if split not in episodes.SPLITS:
            raise ValueError(f"split must be one of {episodes.SPLITS}")
        if render_mode is not None and not cameras:
            raise ValueError("render_mode needs cameras=True")
        self.split = split
        self.cfg: HandConfig = get_hand(hand)
        self.hz = self.cfg.control_hz
        self.cameras = cameras
        self.render_every = render_every
        self.max_ticks = round(max_seconds * self.hz)
        self.visual_randomization = visual_randomization
        self.render_mode = render_mode
        self.reward = Reward(reward)
        self._test = episodes.test_set() if split == "test" else None
        self._renderer: mujoco.Renderer | None = None
        self._images: dict[str, np.ndarray] = {}

        lower = self.cfg.lower().numpy().astype(np.float32)
        upper = self.cfg.upper().numpy().astype(np.float32)
        n = self.cfg.n_dof
        self.action_space = spaces.Box(
            low=np.concatenate([lower, np.zeros(n, np.float32)]),
            high=np.concatenate([upper, np.ones(n, np.float32)]),
            dtype=np.float32)
        obs = {"state": spaces.Box(lower - 5.0, upper + 5.0, dtype=np.float32),
               "prompt": spaces.Text(max_length=128,
                                     charset=string.printable)}
        if cameras:
            size = objects.IMAGE_SIZE
            obs["images"] = spaces.Dict({
                name: spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
                for name in objects.CAMERAS})
        self.observation_space = spaces.Dict(obs)
        self._speed_mm_tick = action_speed_mm_s(self.cfg) / self.hz

    # -- gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        options = options or {}
        if "bottle" in options:
            self.episode = None
            bottle = options["bottle"]
            place = options.get("placement", objects.Placement())
            visual_seed = options.get("visual_seed")
        else:
            index = options.get("index", seed)
            if index is None:
                index = int(self.np_random.integers(
                    len(self._test) if self._test else 2**31))
            self.episode = (self._test[index % len(self._test)] if self._test
                            else episodes.train(index))
            bottle, place = self.episode.bottle, self.episode.placement
            visual_seed = (self.episode.visual_seed
                           if self.visual_randomization else None)
        self._load(bottle, place, visual_seed)
        return self._observe(), self._info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)
        n = self.cfg.n_dof
        goal, effort = action[:n], action[n:]
        self._setpoint += np.clip(goal - self._setpoint,
                                  -self._speed_mm_tick, self._speed_mm_tick)
        self.data.ctrl[:] = self._setpoint / MM_PER_M
        sim.cap_force(self.model, self._full_force_n, effort)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self.monitor(self.model, self.data)
        self._ticks += 1

        reward, terms, failed = self.reward(self._setpoint, self.hz)
        terminated = self.monitor.opened or failed
        truncated = not terminated and self._ticks >= self.max_ticks
        info = self._info()
        info["reward_terms"] = terms
        if terminated or truncated:
            info["metrics"] = self.reward.metrics.as_dict()
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return np.concatenate(list(self._render_views().values()), axis=1)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def episode_metrics(self) -> dict:
        """The current episode's `reward.Metrics` so far, as a dict. What a
        vector env's `call("episode_metrics")` reads when a policy stops before
        the env ends the episode."""
        return self.reward.metrics.as_dict()

    # -- the answer sheet ----------------------------------------------------

    def privileged(self) -> Privileged:
        m = self.monitor
        released = m.released_at_s is not None
        return Privileged(
            bottle=self._bottle, placement=self._place,
            cap_turns=float(self.data.qpos[m._turn] / (2 * np.pi)),
            cap_lift_mm=m._lift_mm(self.data) if released else 0.0,
            released=released, opened=m.opened)

    # -- internals -----------------------------------------------------------

    def _load(self, bottle: objects.Bottle, place: objects.Placement,
              visual_seed: int | str | None) -> None:
        self._bottle, self._place = bottle, place
        # The scene file is only the route into MuJoCo: once loaded it is
        # removed, so parallel workers do not leave one per episode behind.
        path = objects.write_scene(bottle, place)
        self.model, self.data, self._full_force_n = sim.load(path, self.cfg)
        os.remove(path)
        if visual_seed is not None:
            import random
            objects.randomize_visuals(self.model, random.Random(visual_seed))
        if self.cameras:
            self.close()
            self._renderer = mujoco.Renderer(self.model, objects.IMAGE_SIZE,
                                             objects.IMAGE_SIZE)
        self._lead = np.array([row[0] for row in qpos_addrs(self.model)])
        self._substeps = max(1, round((1.0 / self.hz) / self.model.opt.timestep))
        self.monitor = objects.ThreadMonitor(bottle)
        self.monitor(self.model, self.data)
        self._ticks = 0
        self._setpoint = self._position()
        self.reward.reset(self.model, self.data, self.monitor, self._setpoint)

    def _position(self) -> np.ndarray:
        return (self.data.qpos[self._lead] * MM_PER_M).astype(np.float32)

    def _render_views(self) -> dict[str, np.ndarray]:
        images = {}
        for name in objects.CAMERAS:
            self._renderer.update_scene(self.data, name)
            images[name] = self._renderer.render()
        return images

    def _observe(self) -> dict:
        obs = {"state": self._position(), "prompt": PROMPT}
        if self.cameras:
            if self._ticks % self.render_every == 0:
                self._images = self._render_views()
            obs["images"] = self._images
        return obs

    def _info(self) -> dict:
        ep = self.episode
        return {"is_success": self.monitor.opened,
                "elapsed_s": self._ticks / self.hz,
                "episode": (ep.split, ep.index) if ep else None,
                "privileged": self.privileged()}


class Teacher:
    """The scripted `cap` task as a policy over `CapEnv`'s action space.

    The one controller built from `Privileged`, and only at construction: each
    tick it reads `obs["state"]`, as a policy under test does. Its action is the
    profiled setpoint the old sim loop wrote to the actuators, not its raw goal,
    so a demonstration is a smooth target trajectory a VLA can imitate rather
    than a jump to the end of every move.
    """

    def __init__(self, hand: HandConfig, privileged: Privileged):
        """Built per episode, from that episode's `info["privileged"]`."""
        seq = tasks.make("cap", hand, torch.zeros(1, hand.n_dof),
                         cfg=objects.teacher_config(tasks.config("cap"),
                                                    privileged.bottle))
        self.runner = PolicyRunner(seq, hand.control_hz)
        self.hz = hand.control_hz
        self._setpoint: torch.Tensor | None = None

    def __call__(self, obs: dict) -> np.ndarray | None:
        """The next action, or None once the task has finished or failed."""
        if self.runner.finished() or self.runner.failed():
            return None
        mm = torch.from_numpy(obs["state"])[None, :]
        if self._setpoint is None:
            self._setpoint = mm.clone()
        a = self.runner.tick(mm)
        self._setpoint = sim.profile(a.goal_mm, self._setpoint,
                                     a.max_speed_mm_s / self.hz)
        return torch.cat([self._setpoint[0], a.effort_limit[0].clamp(0, 1)]
                         ).numpy().astype(np.float32)


def run_episode(env: CapEnv, seed: int | None = None,
                options: dict | None = None,
                make_policy: Callable[[CapEnv, dict], Callable] = (
                    lambda env, info: Teacher(env.cfg, info["privileged"])),
                on_step: Callable[[dict, np.ndarray], None] | None = None
                ) -> dict:
    """Reset, run the policy until it or the episode ends, and return the last
    `info`, with `metrics` even when the policy stopped first. `on_step` sees
    exactly what a recorder would store: the policy's observation and the
    action it took."""
    obs, info = env.reset(seed=seed, options=options)
    policy = make_policy(env, info)
    while True:
        action = policy(obs)
        if action is None:
            break
        if on_step is not None:
            on_step(obs, action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    info.setdefault("metrics", env.episode_metrics())
    return info


def main(obj: str | None = "water_500ml", split: str = "train",
         index: int | None = None, hand: str = DEFAULT_HAND,
         check: bool = False, cameras: bool = False, render_every: int = 1,
         save_frames: str | None = None, placement_seed: int | None = None,
         visual_seed: int | None = None) -> None:
    """Run one scripted-teacher episode and print its outcome.

    Args:
        obj: a catalog bottle from assets/objects/bottles.json, at the nominal
            pose. Ignored when --index is given.
        split: with --index, which split the episode comes from.
        index: run episode `index` of `split` instead of a catalog bottle.
        hand: which entry in `config.HANDS`.
        check: assert every observation's keys are `OBS_KEYS` and nothing
            privileged.
        cameras: render both policy views and report the cost.
        render_every: with --cameras, render on every Nth tick only.
        save_frames: write every 25th tick's views side by side to this
            directory as PNGs. Implies --cameras.
        placement_seed: with --obj, a random `objects.Placement` from this seed.
        visual_seed: with --obj, redraw lighting and colours from this seed.
    """
    import random
    cameras = cameras or save_frames is not None
    env = CapEnv(split, hand, cameras=cameras, render_every=render_every)
    if index is None:
        place = (objects.Placement() if placement_seed is None else
                 objects.sample_placement(random.Random(placement_seed)))
        options = {"bottle": objects.load(obj), "placement": place,
                   "visual_seed": visual_seed}
        if placement_seed is not None:
            print(f"[env] placement {place}")
    else:
        options = {"index": index}
    if save_frames:
        os.makedirs(save_frames, exist_ok=True)
    count = [0]

    def on_step(obs: dict, action: np.ndarray) -> None:
        if check:
            check_obs(obs)
        if save_frames and count[0] % 25 == 0:
            from PIL import Image
            Image.fromarray(np.concatenate(list(obs["images"].values()), 1)
                            ).save(os.path.join(save_frames, f"{count[0]:05d}.png"))
        count[0] += 1

    t0 = time.perf_counter()
    info = run_episode(env, options=options, on_step=on_step)
    wall = time.perf_counter() - t0
    p = info["privileged"]
    print(env.monitor.summary())
    print(f"[env] {p.bottle.name}: success {info['is_success']}, turns "
          f"{p.cap_turns:.2f}, lift {p.cap_lift_mm:.1f} mm, at "
          f"{info['elapsed_s']:.1f} s")
    ticks = env._ticks
    print(f"[env] {ticks} ticks in {wall:.1f} s wall: {ticks / wall:.0f} ticks/s, "
          f"{ticks / env.hz / wall:.1f}x real time"
          + (f", {len(objects.CAMERAS)} views at {objects.IMAGE_SIZE}px"
             if env.cameras else ""))
    if check:
        print(f"[env] check: {count[0]} observations, keys "
              f"{sorted(env.observation_space.spaces)}, none privileged")
    env.close()


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.env")
