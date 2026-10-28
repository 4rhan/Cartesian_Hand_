"""Many `CapEnv`s at once, one process per core, for RL and for evaluation.

    python -m cartesian_hand.vec --bench                     # steps/s against workers
    python -m cartesian_hand.vec --bench --cameras           # the same, rendering both views
    python -m cartesian_hand.vec --eval test --n-envs 16     # scripted teacher on the test split

    envs = make_vec_env(16, split="train", cameras=False)    # RL: autoreset
    results = evaluate(policy, split="test", n_envs=16)      # fixed episodes, once each

`make_vec_env` is a plain `gymnasium.vector.AsyncVectorEnv`: observations
come back batched (`obs["state"]` [N, 7], `obs["images"]["base"]`
[N, 224, 224, 3]), actions go in as [N, 14], and any library that speaks the
gymnasium vector API can drive it. Each worker is its own process, so physics
and rendering run on N cores while the policy -- a VLA on the GPU, or an RL
actor -- runs batched in the parent.

**Spawn, not fork.** An OpenGL context does not survive `fork`, and torch's
thread pool does not either; a spawned worker builds both fresh. Each worker
also pins torch to one thread: the env does tiny tensor ops, and sixteen
workers each starting a pool sized for the whole machine thrash it.

**Evaluation runs a fixed list once.** Autoreset would hand each finished
env whatever episode its own generator drew next, so which bottles get run
-- and how often -- would depend on N. `evaluate` turns autoreset off and
resets each finished env onto the next index in the list by seed (`reset(seed=i)`
is episode `i`), so 50 test episodes are those 50, once each, for any N.
"""
import contextlib
import io
import os
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
import tyro
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .config import DEFAULT_HAND, HandConfig, get_hand
from .env import CapEnv, Teacher


class _Make:
    """Picklable env constructor for a spawned worker."""

    def __init__(self, split: str, env_kwargs: dict):
        self.split, self.env_kwargs = split, env_kwargs

    def __call__(self) -> CapEnv:
        torch.set_num_threads(1)
        return CapEnv(self.split, **self.env_kwargs)


def make_vec_env(n_envs: int, split: str = "train",
                 autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP,
                 **env_kwargs) -> AsyncVectorEnv:
    """`n_envs` `CapEnv`s on `split`, each in its own spawned process.

    `env_kwargs` go to every `CapEnv` (`cameras`, `render_every`, `hand`, ...).
    Seed with `envs.reset(seed=s)`: env i starts on episode `s + i`, then draws
    its own. For headless rendering on a GPU box, set `MUJOCO_GL=egl` before
    calling; workers inherit it.
    """
    return AsyncVectorEnv([_Make(split, env_kwargs) for _ in range(n_envs)],
                          context="spawn", autoreset_mode=autoreset_mode)


class VecPolicy(Protocol):
    """A batched policy `evaluate` can drive."""

    def reset(self, env_ids: np.ndarray, infos: dict) -> None:
        """Envs `env_ids` just started an episode; `infos` is the vector info."""

    def __call__(self, obs: dict) -> np.ndarray:
        """[N, 14] actions for the batched observation."""

    gave_up: np.ndarray
    """[N] bool: this env's policy has stopped. `evaluate` ends the episode
    rather than let it run out its time. Optional: a policy without it never
    gives up."""

    def stop(self, env_ids: np.ndarray) -> None:
        """Optional: envs `env_ids` have no episode left. Their actions are
        ignored from here on, so the policy can drop their state."""


class VecTeacher:
    """One scripted `Teacher` per env, built from that env's answer sheet.

    The teacher's tensor bookkeeping runs here in the parent, one env after
    another, and costs several times a tick's physics -- so it scales worse
    with N than a batched network would. `episodes.evaluate_teacher`, which
    runs whole episodes per process, is the faster way to score the teacher
    alone; this is the reference `VecPolicy`, and the shape a residual policy
    wrapping a base policy takes.
    """

    def __init__(self, hand: HandConfig):
        self.hand = hand
        self.teachers: list[Teacher | None] = []
        self.gave_up = np.zeros(0, dtype=bool)

    def reset(self, env_ids: np.ndarray, infos: dict) -> None:
        grow = int(env_ids.max()) + 1 - len(self.teachers)
        if grow > 0:
            self.teachers += [None] * grow
            self.gave_up = np.concatenate([self.gave_up, np.zeros(grow, bool)])
        for i in env_ids:
            self.teachers[i] = Teacher(self.hand, infos["privileged"][i])
            self.gave_up[i] = False

    def stop(self, env_ids: np.ndarray) -> None:
        for i in env_ids:
            self.teachers[i] = None

    def __call__(self, obs: dict) -> np.ndarray:
        n = self.hand.n_dof
        actions = np.zeros((len(self.teachers), 2 * n), dtype=np.float32)
        for i, teacher in enumerate(self.teachers):
            # The task's per-row log is for one hand on a bench; N of them
            # interleaved say nothing. Outcomes come from the metrics instead.
            with contextlib.redirect_stdout(io.StringIO()):
                action = teacher({"state": obs["state"][i]}) if teacher else None
            if action is None:
                self.gave_up[i] = True
                action = np.concatenate([obs["state"][i],
                                         np.zeros(n, np.float32)])
            actions[i] = action
        return actions


def evaluate(policy: VecPolicy, split: str = "test",
             episodes: Sequence[int] | None = None, n_envs: int = 8,
             verbose: bool = True, **env_kwargs) -> list[dict]:
    """Run `policy` once on each episode index in `episodes` (default: the
    whole test split, or the first 50 train episodes), `n_envs` at a time.

    Returns one dict per episode, in `episodes` order: its index plus every
    `reward.Metrics` field.
    """
    from . import episodes as E
    if episodes is None:
        episodes = range(len(E.test_set()) if split == "test" else 50)
    queue = list(episodes)
    n_envs = min(n_envs, len(queue))
    envs = make_vec_env(n_envs, split, autoreset_mode=AutoresetMode.DISABLED,
                        **env_kwargs)
    running = np.array(queue[:n_envs])
    queue = queue[n_envs:]
    results: dict[int, dict] = {}
    t0, steps = time.perf_counter(), 0
    try:
        obs, infos = envs.reset(seed=[int(i) for i in running])
        policy.reset(np.arange(n_envs), infos)
        while (running >= 0).any():
            actions = policy(obs)
            idle = running < 0
            # An env with nothing left to run keeps stepping (the vector env
            # steps all N); hold it still rather than act on a finished scene.
            actions[idle] = np.concatenate(
                [obs["state"][idle], np.zeros_like(obs["state"][idle])], axis=1)
            obs, _, terminated, truncated, infos = envs.step(actions)
            steps += int((running >= 0).sum())
            gave_up = getattr(policy, "gave_up", np.zeros(n_envs, bool))
            ended = (terminated | truncated | gave_up) & (running >= 0)
            if not ended.any():
                continue
            metrics = envs.call("episode_metrics")
            restart = []
            for i in np.flatnonzero(ended):
                results[int(running[i])] = {"index": int(running[i]),
                                            **metrics[i]}
                if verbose:
                    _print_episode(split, results[int(running[i])])
                running[i] = queue.pop(0) if queue else -1
                if running[i] >= 0:
                    restart.append(i)
                elif hasattr(policy, "stop"):
                    policy.stop(np.array([i]))
            if restart:
                mask = np.zeros(n_envs, dtype=bool)
                mask[restart] = True
                seeds = [int(running[i]) if mask[i] else None
                         for i in range(n_envs)]
                obs, infos = envs.reset(seed=seeds,
                                        options={"reset_mask": mask})
                policy.reset(np.array(restart), infos)
    finally:
        envs.close()
    wall = time.perf_counter() - t0
    ordered = [results[i] for i in episodes]
    if verbose:
        summarize(ordered, split)
        print(f"[vec] {steps} env steps in {wall:.1f} s on {n_envs} workers: "
              f"{steps / wall:.0f} steps/s")
    return ordered


def _print_episode(split: str, r: dict) -> None:
    outcome = ("OPENED" if r["success"] else
               "dropped" if r["dropped"] else "tipped" if r["tipped"] else
               "released, not lifted" if r["released"] else
               f"turned {r['turn_fraction']:.0%}")
    print(f"  {split} {r['index']:3d}  {outcome:22s} return "
          f"{r['episode_return']:+6.2f}  {r['ticks']:4d} ticks")


def summarize(results: list[dict], split: str) -> dict:
    """Success rate and the failure breakdown over `results`."""
    n = len(results)
    rate = lambda key: sum(bool(r[key]) for r in results) / n
    won = [r["time_to_success_s"] for r in results if r["success"]]
    summary = {"episodes": n, "success": rate("success"),
               "released": rate("released"), "dropped": rate("dropped"),
               "tipped": rate("tipped"),
               "mean_return": float(np.mean([r["episode_return"] for r in results])),
               "mean_time_to_success_s": float(np.mean(won)) if won else float("nan")}
    print(f"[vec] {split}: success {summary['success']:.0%} of {n}, released "
          f"{summary['released']:.0%}, dropped {summary['dropped']:.0%}, tipped "
          f"{summary['tipped']:.0%}, mean return {summary['mean_return']:+.2f}, "
          f"mean time to success {summary['mean_time_to_success_s']:.1f} s")
    return summary


def benchmark(workers: Sequence[int], steps: int = 200, **env_kwargs) -> None:
    """Env steps/s for each worker count, holding every joint where it is --
    the physics and rendering cost alone, no policy in the way."""
    for n in workers:
        envs = make_vec_env(n, **env_kwargs)
        obs, _ = envs.reset(seed=0)
        hold = lambda o: np.concatenate(
            [o["state"], np.full_like(o["state"], 0.3)], axis=1)
        for _ in range(5):                       # warm up the pipes
            obs, *_ = envs.step(hold(obs))
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, *_ = envs.step(hold(obs))
        wall = time.perf_counter() - t0
        envs.close()
        rate = n * steps / wall
        print(f"[vec] {n:3d} workers: {rate:7.0f} env steps/s, "
              f"{rate / 50:5.1f}x real time in total")


def main(eval: str | None = None, n_envs: int = os.cpu_count() or 1,
         n: int | None = None, bench: bool = False,
         bench_workers: tuple[int, ...] = (),
         cameras: bool = False, render_every: int = 1,
         hand: str = DEFAULT_HAND) -> None:
    """Benchmark the vector env, or evaluate the scripted teacher through it.

    Args:
        eval: `train` or `test`: run the scripted teacher on that split.
        n_envs: worker processes.
        n: with --eval, the first n episodes (default all 50 test, or 50 train).
        bench_workers: with --bench, the worker counts to time (default 1, 2,
            4, 8 and every core).
        bench: time env steps/s against worker count.
        cameras: render both policy views in every env.
        render_every: with --cameras, render on every Nth tick only.
        hand: which entry in `config.HANDS`.
    """
    torch.set_num_threads(1)
    env_kwargs = dict(cameras=cameras, render_every=render_every, hand=hand)
    if bench:
        cores = os.cpu_count() or 1
        benchmark(bench_workers or sorted({1, 2, 4, 8, cores}), **env_kwargs)
    if eval is not None:
        episodes = range(n) if n is not None else None
        evaluate(VecTeacher(get_hand(hand)), eval, episodes, n_envs,
                 **env_kwargs)


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.vec")
