"""Record scripted-teacher demonstrations as a LeRobot v3.0 dataset, for pi0.5.

    python -m cartesian_hand.record --out data/cap_demos --n 200          # 200 train episodes
    python -m cartesian_hand.record --out data/cap_demos --n 4 --workers 2   # a quick look
    python -m cartesian_hand.record --replay data/cap_demos --episode 0   # replay one

One frame per control tick (50 Hz), holding exactly what the policy under test
sees and does -- `CapEnv`'s observation and the action taken on it:

    observation.images.base    video  [224, 224, 3]   fixed camera on the jaws
    observation.images.wrist   video  [224, 224, 3]   camera riding the bridge
    observation.state          float32 [7]   joint positions, mm, DOF order
    action                     float32 [14]  7 position goals (mm), 7 efforts (0-1)
    task                       the prompt, `env.PROMPT`
    next.reward / next.done / next.success   float32 / bool [1], from the step

The action is the teacher's profiled setpoint, not its raw goal: replayed
through `CapEnv.step` it drives the same motion, which `--replay` checks.

**Only the train split.** Demonstrations of test bottles would put the held-out
set in the training data, so the recorder has no `--split`.

**Nothing privileged in a frame.** The bottle's measurements, placement and the
cap's state stay out of the features, as they stay out of `obs`. What each
episode was -- its train index, the bottle, its metrics -- goes in a sidecar,
`meta/cartesian_hand.json`, for filtering and for an RL critic, never as an
input key a training config could pick up by accident.

**Failures are dropped** unless `--keep-failures`: the teacher opens ~94% of
bottles, and imitating its other 6% teaches the policy to fail.

**Parallel by shards.** Each worker process records its own range of train
indices into a dataset of its own under `<out>/.shards`, and lerobot's
`aggregate_datasets` merges them, in index order, into `<out>`. Rendering is
the cost (two views a tick, ~23 ms on this laptop's CPU), so workers scale it.
"""
import contextlib
import io
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import tyro

from . import objects
from .config import DEFAULT_HAND, LABELS, HandConfig, get_hand
from .env import PROMPT, CapEnv, Teacher

ROBOT_TYPE = "cartesian_hand"
SIDECAR = "meta/cartesian_hand.json"


def _quiet() -> None:
    """Silence the AV1 encoder's banner and the datasets progress bars, which
    per episode and per worker bury the summary lines. (libav's two-line mp4
    note stays: lerobot restores libav's own logger after every encode.)"""
    os.environ.setdefault("SVT_LOG", "1")        # errors only; read at encoder start
    import datasets
    datasets.disable_progress_bars()


def _snake(label: str) -> str:
    return label.replace(" ", "_")


def features(hand: HandConfig) -> dict:
    """The dataset's frame layout. Names travel into `meta/info.json`, so a
    training config can map `observation.state` without reading this file."""
    joints = [_snake(l) for l in LABELS[:hand.n_dof]]
    size = objects.IMAGE_SIZE
    feats = {f"observation.images.{name}": {
                 "dtype": "video", "shape": (size, size, 3),
                 "names": ["height", "width", "channels"]}
             for name in objects.CAMERAS}
    feats["observation.state"] = {"dtype": "float32", "shape": (hand.n_dof,),
                                  "names": [f"{j}_mm" for j in joints]}
    feats["action"] = {"dtype": "float32", "shape": (2 * hand.n_dof,),
                       "names": [f"{j}_goal_mm" for j in joints]
                                + [f"{j}_effort" for j in joints]}
    for key, dtype in (("next.reward", "float32"), ("next.done", "bool"),
                       ("next.success", "bool")):
        feats[key] = {"dtype": dtype, "shape": (1,), "names": None}
    return feats


def _record_shard(job: tuple) -> list[dict]:
    """Record train episodes `indices` into a fresh dataset at `root`. Returns
    one summary per kept episode, in recording order."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root, indices, hand_name, keep_failures, max_seconds = job
    torch.set_num_threads(1)
    _quiet()
    hand = get_hand(hand_name)
    env = CapEnv("train", hand_name, cameras=True, render_every=1,
                 max_seconds=max_seconds)
    ds = LeRobotDataset.create(repo_id=f"local/{Path(root).name}", root=root,
                               fps=round(env.hz), features=features(hand),
                               robot_type=ROBOT_TYPE, use_videos=True,
                               streaming_encoding=True)
    kept = []
    for index in indices:
        t0 = time.perf_counter()
        obs, info = env.reset(seed=index)
        teacher = Teacher(hand, info["privileged"])
        ticks = 0
        # The task's per-row log is for one hand on a bench; here it would
        # interleave across workers. The summary line below replaces it.
        with contextlib.redirect_stdout(io.StringIO()):
            while (action := teacher(obs)) is not None:
                frame = {f"observation.images.{k}": v
                         for k, v in obs["images"].items()}
                frame["observation.state"] = obs["state"]
                frame["action"] = action
                frame["task"] = obs["prompt"]
                obs, reward, terminated, truncated, info = env.step(action)
                frame["next.reward"] = np.array([reward], np.float32)
                frame["next.done"] = np.array([terminated or truncated])
                frame["next.success"] = np.array([info["is_success"]])
                ds.add_frame(frame)
                ticks += 1
                if terminated or truncated:
                    break
        metrics = env.episode_metrics()
        ep = env.episode
        keep = (metrics["success"] or keep_failures) and ticks > 0
        if keep:
            ds.save_episode()
            kept.append({"train_index": index, "bottle": ep.bottle.name,
                         "frames": ticks, **metrics})
        else:
            ds.clear_episode_buffer()
        print(f"  train {index:4d}  {ep.bottle.name:26s} "
              f"{'OPENED' if metrics['success'] else 'failed':7s} {ticks:4d} frames  "
              f"{'kept' if keep else 'dropped'}  ({time.perf_counter() - t0:.1f} s)",
              flush=True)
    ds.finalize()
    env.close()
    return kept


def record(out: str, n: int, start: int = 0, workers: int = 1,
           hand: str = DEFAULT_HAND, keep_failures: bool = False,
           max_seconds: float = 30.0, repo_id: str = "local/cartesian_hand_cap",
           force: bool = False) -> Path:
    """Record train episodes `start .. start+n-1` into a LeRobot dataset at
    `out`. Returns its root."""
    from lerobot.datasets.aggregate import aggregate_datasets
    _quiet()
    out = Path(out).resolve()
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    shard_dir = out.parent / f".{out.name}.shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True)

    indices = list(range(start, start + n))
    workers = max(1, min(workers, n))
    # Contiguous ranges, so concatenating shards in order keeps index order.
    bounds = np.linspace(0, n, workers + 1).round().astype(int)
    jobs = [(str(shard_dir / f"shard-{k:03d}"), indices[bounds[k]:bounds[k + 1]],
             hand, keep_failures, max_seconds) for k in range(workers)]

    t0 = time.perf_counter()
    print(f"[record] train {start}..{start + n - 1} on {workers} workers -> {out}")
    # Spawn: an OpenGL context and torch's thread pool do not survive fork.
    with ProcessPoolExecutor(workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        shards = list(pool.map(_record_shard, jobs))

    used = [(job[0], kept) for job, kept in zip(jobs, shards) if kept]
    if not used:
        shutil.rmtree(shard_dir)
        raise RuntimeError("no episode was kept; nothing to write")
    aggregate_datasets(repo_ids=[f"local/{Path(r).name}" for r, _ in used],
                       aggr_repo_id=repo_id, roots=[Path(r) for r, _ in used],
                       aggr_root=out)
    shutil.rmtree(shard_dir)

    episodes = [dict(episode_index=i, **e)
                for i, e in enumerate(e for _, kept in used for e in kept)]
    with open(out / SIDECAR, "w") as f:
        json.dump({"_comment": "Per-episode provenance, written by "
                               "cartesian_hand.record. Privileged: not a policy "
                               "input.",
                   "split": "train", "prompt": PROMPT, "hand": hand,
                   "attempted": indices, "keep_failures": keep_failures,
                   "episodes": episodes}, f, indent=1)

    frames = sum(e["frames"] for e in episodes)
    wall = time.perf_counter() - t0
    print(f"[record] kept {len(episodes)}/{n} episodes, {frames} frames "
          f"({frames / get_hand(hand).control_hz / 60:.1f} min of demonstration) "
          f"in {wall:.0f} s")
    print(f"[record] wrote {out}")
    return out


def replay_episode(root: str, episode: int = 0, hand: str = DEFAULT_HAND) -> bool:
    """Load `root` with lerobot, drive `CapEnv` with episode `episode`'s
    recorded actions on the bottle it was recorded on, and report whether the
    cap opens again. Proves the dataset round-trips: v3 files lerobot can read,
    and actions that mean the same thing in the env."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root).resolve()
    ds = LeRobotDataset(repo_id="local/replay", root=root, episodes=[episode])
    with open(root / SIDECAR) as f:
        meta = json.load(f)["episodes"][episode]
    print(f"[replay] {root.name}: {ds.meta.info.codebase_version}, "
          f"{ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
          f"{ds.fps} fps, features {sorted(ds.features)}")
    frame = ds[0]
    print(f"[replay] episode {episode} = train {meta['train_index']} "
          f"({meta['bottle']}), {len(ds)} frames; frame 0: "
          + ", ".join(f"{k} {tuple(v.shape)}" for k, v in frame.items()
                      if hasattr(v, "shape") and v.ndim)
          + f", task {frame['task']!r}")

    actions = np.stack([a.numpy() for a in ds.hf_dataset["action"]])
    env = CapEnv("train", hand, cameras=False)
    env.reset(seed=meta["train_index"])
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    p = info["privileged"]
    print(f"[replay] replayed {len(actions)} actions: success {info['is_success']}, "
          f"turns {p.cap_turns:.2f}, lift {p.cap_lift_mm:.1f} mm "
          f"(recorded: success {meta['success']})")
    return bool(info["is_success"])


def main(out: str | None = None, n: int = 10, start: int = 0,
         workers: int = os.cpu_count() or 1, hand: str = DEFAULT_HAND,
         keep_failures: bool = False, max_seconds: float = 30.0,
         repo_id: str = "local/cartesian_hand_cap", force: bool = False,
         replay: str | None = None, episode: int = 0) -> None:
    """Record teacher demonstrations as a LeRobot v3.0 dataset, or replay one.

    Args:
        out: directory to write the dataset to.
        n: train episodes to attempt.
        start: first train index; `--start 1000 --n 500` records a disjoint set.
        workers: parallel recording processes.
        hand: which entry in `config.HANDS`.
        keep_failures: keep episodes the teacher failed, too.
        max_seconds: simulated time before an episode is cut off.
        repo_id: the dataset's name in its metadata, `user/name` for the Hub.
        force: replace `out` if it exists.
        replay: a recorded dataset: load it and replay `--episode`'s actions.
        episode: with --replay, which episode.
    """
    torch.set_num_threads(1)
    if replay is not None:
        replay_episode(replay, episode, hand)
        return
    if out is None:
        raise SystemExit("pass --out DIR to record, or --replay DIR")
    record(out, n, start, workers, hand, keep_failures, max_seconds, repo_id,
           force)


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.record")
