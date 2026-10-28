# Policy episodes

`cartesian_hand/env.py` is the cap-opening task as a
[gymnasium](https://gymnasium.farama.org) environment, `CapEnv`. The same env
serves a VLA (pi0.5) being tested or recorded and an RL algorithm being
trained, on the same physics `sim` uses.

```bash
pip install -e .[rl]                                      # gymnasium
python -m cartesian_hand.env --obj water_500ml            # one scripted-teacher episode
python -m cartesian_hand.env --split test --index 3       # test episode 3
python -m cartesian_hand.env --obj water_500ml --check    # also prove nothing leaks
```

```python
from cartesian_hand.env import CapEnv, Teacher
env = CapEnv(split="train", cameras=True)
obs, info = env.reset(seed=0)            # train episode 0
teacher = Teacher(env.cfg, info["privileged"])   # scripted policy, from the answer sheet
obs, reward, terminated, truncated, info = env.step(teacher(obs))
info["is_success"]                       # cap lifted
```

## Observation and action

| | Shape | Meaning |
|---|---|---|
| `obs["state"]` | float32 [7] | joint positions, mm, in DOF order |
| `obs["images"]["base"]`, `["wrist"]` | uint8 [224, 224, 3] | the two policy cameras (with `cameras=True`) |
| `obs["prompt"]` | str | `"unscrew the cap and lift it off the bottle"` |
| `action[:7]` | float32 | position goals, mm, within the hand's travel |
| `action[7:]` | float32 | effort limits, 0-1 (servo `torque / 1000`) |

Each joint's setpoint walks toward its goal at no more than the servo's top
speed (84 mm/s, `env.action_speed_mm_s`), as the servo's own profiler would.
To move slower, send nearer goals. The action space is in physical units, not
normalized to [-1, 1]: a VLA's action head and the bench both speak mm and
effort, and an RL wrapper can rescale it.

## What the policy sees, and what it doesn't

| `obs` (the policy under test) | `info["privileged"]` (teacher, scorer, RL critic) |
|---|---|
| joint positions | the bottle's measurements (`Bottle`) and `Placement` |
| camera images | `cap_turns`: how far the cap has turned |
| prompt | `cap_lift_mm`, `released`, `opened` |

Everything on the left exists on the bench. The right-hand side is the answer:
a policy that knew `turns_to_open` or `cap_offset` would succeed in sim for a
reason the bench doesn't have.

- The **scripted teacher** (`env.Teacher`, the `cap` task) is the one
  controller built from `Privileged`, through `objects.teacher_config`, before
  the episode starts. Each tick it reads only `obs["state"]`. Its action is
  the smooth setpoint it would have sent, not a jump to each move's end, so
  demonstrations are something a VLA can imitate.
- An **RL critic** may read `info["privileged"]` too (asymmetric actor-critic):
  it is only needed in training, and only the actor is deployed.
- `check_obs(obs)` fails if an observation has any key outside `state`,
  `images`, `prompt`, or one named like a privileged field. `--check` runs it
  on every observation of an episode.

## Episodes

`reset(seed=s)` gives episode `s` of the env's split: train episode `s`, or
test episode `s mod 50`. `reset()` with no seed draws one from the env's own
random generator. `options={"index": i}` picks one explicitly, and
`options={"bottle": b, "placement": p, "visual_seed": v}` runs a bottle that
is in no split. An episode is `terminated` when the cap is lifted (success)
and `truncated` after `max_seconds` (30 s).

## Reward and metrics

`cartesian_hand/reward.py`. `info["reward_terms"]` holds each term every step;
`info["metrics"]` holds the episode's outcome on its last step.

| Term | Weight (`RewardConfig`) | Paid |
|---|---|---|
| `progress` | `turn` 1.0, `lift` 1.0 | the change in turn progress (0 to `turns_to_open`) plus lift progress (0 to 10 mm), every tick |
| `release` | 1.0 | once, when the thread lets go |
| `success` | 5.0 | once, when the cap is lifted clear; ends the episode |
| `drop` | -5.0 | once, if the released cap lands on the table or falls off the neck; ends the episode |
| `tip` | -5.0 | once, if the bottle leans past 20 degrees; ends the episode |
| `jerk` | -0.001 per mm^2/tick^6 | the squared third difference of the commanded setpoints, every tick |

Progress is potential-based (reward = change in a potential), so it guides
exploration without changing which policy is optimal. That matters for
residual RL on top of a VLA: the reward shouldn't pull the residual toward
anything but opening the cap. Turning back loses exactly what turning forward
earned, so dithering pays nothing.

Measured with the scripted teacher on `water_500ml`: progress +2.0, release
+1.0, success +5.0, jerk -0.19, return **+7.82**. A policy flipping its goals
at full speed every tick would pay about -1.3 in jerk *per tick*.

`info["metrics"]`: `success`, `time_to_success_s`, `released`,
`turn_fraction` (best progress toward `turns_to_open`), `max_lift_mm`,
`dropped`, `tipped`, `episode_return`, `jerk_rms`, `ticks`. Failures say
*why*: a cap that turned 0.98 of the way, one that released but wasn't lifted,
and one that was dropped all score 0 success but different metrics.

```bash
python -m cartesian_hand.reward --obj water_500ml          # teacher's terms and metrics
python -m cartesian_hand.reward --split test --index 15    # a failing episode
```

## Placement randomization

`objects.Placement` stands the bottle the way a person setting it down would:

| Field | `sample_placement` range |
|---|---|
| `dx_mm`, `dy_mm` | +/-3 mm off centre, along the fingers and across the jaws |
| `yaw_deg` | any rotation about the bottle's axis |
| `tilt_deg`, `tilt_dir_deg` | up to 2 degrees, leaning in any direction |

The tilt belongs to the table, not the bottle: a bottle tilted on a level table
just rocks back flat, so the table is tilted under it and the bottle stands
square on it. Ranges are in `objects.PLACEMENT_RANGE`.

```bash
python -m cartesian_hand.objects water_500ml --view --placement-seed 3   # look at one
python -m cartesian_hand.env --obj water_500ml --placement-seed 3       # run the teacher on it
```

Measured: a placed bottle settles to within 0.01 degrees of the requested tilt
with no drift, and the teacher opened 8/8 random placements (seeds 0-7).

## Train and test splits

`cartesian_hand/episodes.py` fixes which bottle, standing where, each episode
gets. An episode is a variant of `water_500ml` or `pill_bottle` plus a
`Placement`, drawn from ranges wider than `objects.vary`'s defaults
(`episodes.HARD`): sizes, masses, pitch, turns and friction +/-35%, resisting
torque over a factor of 3 either way, +/-5 mm off centre, up to 3 degrees of tilt.

- **train**: `episodes.train(i)`, drawn on demand from seed `"train:<i>"`. Any
  index is valid.
- **test**: `episodes.test_set()`, 50 episodes drawn once from seed
  `"test:<i>"` and frozen in `assets/objects/splits/test.json`, so a later
  change to the ranges can't quietly swap the test bottles.
  `--write-test` refuses to overwrite it without `--force`.

```bash
python -m cartesian_hand.episodes                       # describe both splits
python -m cartesian_hand.episodes --eval test --n 20    # teacher success, in parallel
```

Measured: the scripted teacher opens 47/50 test episodes (94%). The three
misses are small or high-torque caps that slip just short of release.
That is the ceiling a policy imitating the teacher should be expected to reach.

## Opening-only episodes

An episode ends when the cap is **lifted**: released from the thread and
raised 10 mm (`ThreadMonitor.OPENED_LIFT_MM`) above the neck's top, along the
bottle's axis. That is the score, `Privileged.opened`. The closing half isn't
modelled, so it is skipped, and an episode takes ~12 s of simulated time
instead of ~22. `sim --obj` stops at the same point unless given `--full-cycle`.

## Policy cameras

Two views at 224 x 224, pi0.5's image input size (openpi resizes every view
to 224, so rendering bigger only costs time). Both are defined in
`objects.CAMERAS` and added to every object scene:

| Camera | Mounted | openpi slot | Sees |
|---|---|---|---|
| `base` | fixed, past the fingertips, looking in between the jaws | `base_0_rgb` | the cap between the aux pads, the bottle, the whole hand |
| `wrist` | on the z stage, so it moves with the aux jaw | `left_wrist_0_rgb` | the cap centred between the jaws, close up |

```bash
python -m cartesian_hand.env --obj water_500ml --cameras                      # time it
python -m cartesian_hand.env --obj water_500ml --save-frames /tmp/frames      # every 25th tick as PNGs
python -m cartesian_hand.env --obj water_500ml --cameras --render-every 5     # images at 10 Hz
```

**Rendering speed**, measured on the dev machine (no NVIDIA GPU, `MUJOCO_GL=glfw`;
`egl` was unavailable there):

| | per tick | episode (590 ticks) |
|---|---|---|
| physics only | 1.7 ms | 114 ticks/s, 2.3x real time |
| + 2 views every tick | +23 ms (11.4 ms a view) | 28 ticks/s, 0.6x real time |
| + 2 views every 5th tick (`render_every=5`) | +4.6 ms average | 68 ticks/s, 1.4x real time |

A view costs the same at 448 px as at 224 (12.7 against 11.4 ms), so the cost is
the GL round trip, not pixels. A policy only reads images when it runs
inference, so render at its inference rate. On a GPU box run headless with
`MUJOCO_GL=egl` and re-measure.

## Visual randomization

`reset(options={..., "visual_seed": v})` redraws lighting and colours
(`objects.randomize_visuals`): light intensity, position and direction, the
headlight, the cap's and bottle's colours (and the bottle's transparency), the
table's colour, and a +/-30% shade on the hand's parts. Each episode from
`episodes.py` has its own seed, `Episode.visual_seed`, derived from its split
and index, so the frozen test file didn't need to change. `CapEnv` applies it
unless `visual_randomization=False`.

Only render-side fields change: an episode ends with bit-identical joint
positions with or without it. A policy whose answer changes when the cap turns
red has learned the colour, not the cap.

```bash
python -m cartesian_hand.env --obj water_500ml --visual-seed 3 --save-frames /tmp/frames
```

## Parallel envs

`cartesian_hand/vec.py` runs N `CapEnv`s, one spawned process each, as a
standard `gymnasium.vector.AsyncVectorEnv`: observations come back batched
(`obs["state"]` [N, 7], `obs["images"]["base"]` [N, 224, 224, 3]) and actions
go in as [N, 14]. Physics and rendering use N cores while the policy (a VLA on
the GPU, or an RL actor) runs batched in the parent.

```python
from cartesian_hand.vec import make_vec_env, evaluate, VecTeacher
envs = make_vec_env(16, split="train", cameras=False)   # RL: autoreset, env i starts on episode seed+i
obs, infos = envs.reset(seed=0)

results = evaluate(my_policy, split="test", n_envs=16)  # each of the 50 test episodes, once
```

```bash
python -m cartesian_hand.vec --bench                    # env steps/s against worker count
python -m cartesian_hand.vec --bench --cameras --render-every 5
python -m cartesian_hand.vec --eval test --n-envs 16    # scripted teacher through the vector env
```

- **Training** uses gymnasium's default autoreset: a finished env starts
  its next episode on the following `step`.
- **Evaluation** turns autoreset off and resets each finished env onto the
  next index in the list, by seed. So the 50 test episodes are exactly those
  50, once each, whatever N is. A policy for `evaluate` implements
  `VecPolicy`: `reset(env_ids, infos)` when envs start an episode,
  `__call__(obs) -> [N, 14]`, and optionally `gave_up` and `stop(env_ids)`.
  `VecTeacher` is the reference implementation, and the shape a residual
  policy wrapping a base policy takes.
- Workers are **spawned**, not forked (an OpenGL context and torch's thread
  pool don't survive fork), and each pins torch to one thread. On a headless
  GPU machine, set `MUJOCO_GL=egl` before creating the envs; workers inherit it.

**The scripted teacher doesn't scale here.** Its per-tick tensor bookkeeping
(~12 ms) runs in the parent, one env after another, so with 16 envs each
vector step waits ~0.2 s on it. A batched network policy pays that once per
step, not N times. To score the teacher alone, `episodes --eval` (whole
episodes per process) is much faster.

**Measured** on the dev laptop (AMD Ryzen 7 7735HS: 8 cores / 16 threads,
integrated Radeon graphics, `MUJOCO_GL=glfw`), with other work loading it
(load average ~4). Env steps/s, every joint held still:

| Workers | No cameras | 2 views, `render_every=5` |
|---|---|---|
| 1 | 84 (1.7x real time) | 41 (0.8x) |
| 2 | 173 | |
| 4 | 272 | 104 |
| 8 | 429 | 131 |
| 16 | 514 (10.3x real time) | 160 (3.2x) |

Physics scales close to linearly up to the 8 physical cores and flattens past
them; rendering flattens sooner, sharing one integrated GPU. On a
workstation with a discrete GPU and `MUJOCO_GL=egl`, measure again.

`python -m cartesian_hand.vec --eval test --n-envs 16` gives the same 47/50
(94%) as `episodes --eval`, with mean return +7.31. It takes 14 minutes, against
about a minute for `episodes --eval`, because of the teacher's serial cost above.

## Recording demonstrations

`cartesian_hand/record.py` runs the scripted teacher over train episodes and
writes what the policy saw and did as a [LeRobot](https://github.com/huggingface/lerobot)
**v3.0** dataset, the format pi0.5 is fine-tuned from.

```bash
pip install -e .[record]                                             # lerobot[dataset]
python -m cartesian_hand.record --out data/cap_demos --n 500 --workers 8
python -m cartesian_hand.record --replay data/cap_demos --episode 0  # load it back, replay it
```

One frame per control tick, 50 fps:

| Feature | Type | What |
|---|---|---|
| `observation.images.base` | video 224×224×3 | fixed camera on the jaws |
| `observation.images.wrist` | video 224×224×3 | camera on the bridge |
| `observation.state` | float32 [7] | joint positions, mm, DOF order |
| `action` | float32 [14] | 7 position goals (mm), then 7 efforts (0–1) |
| `task` | text | the prompt |
| `next.reward`, `next.done`, `next.success` | [1] | from the step the action caused |

- **Exactly the policy's interface.** Each frame is `CapEnv`'s `obs` and the
  action taken on it. The action is the teacher's profiled setpoint, so
  replaying it through `CapEnv.step` repeats the motion; `--replay` loads the
  dataset with lerobot and checks that the cap opens again.
- **Train split only.** There is no `--split`: demonstrations of test bottles
  would leak the held-out set into training. `--start` picks a disjoint range
  for a second batch.
- **Nothing privileged in a frame.** The bottle, its placement and the cap's
  state go in a sidecar, `meta/cartesian_hand.json`: per episode, its train
  index, bottle, frame count and `reward.Metrics`. Use it to filter episodes
  or to rebuild the answer sheet for an RL critic, never as a policy input.
- **Failed episodes are dropped** unless `--keep-failures`. The teacher opens
  about 94% of bottles, and imitating its failures teaches failing.
- **Parallel by shards.** Each worker records a contiguous range of indices
  into its own dataset, and lerobot's `aggregate_datasets` merges them in
  index order. Videos are AV1 (lerobot's default), encoded while recording.

**Measured** on the dev laptop (8 cores, integrated graphics, `MUJOCO_GL=glfw`,
load average ~8): 16 episodes on 8 workers kept 14 (8,358 frames, 2.8 minutes
of demonstration) in 200 s, about 14 s and 3.6 MB per kept episode. Rendering
two views per tick is the cost, so 500 episodes take about 2 hours and 1.8 GB
here; a GPU machine with `MUJOCO_GL=egl` should be much faster.
