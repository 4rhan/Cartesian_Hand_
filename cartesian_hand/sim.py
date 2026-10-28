"""The other backend: run a task against mujoco instead of against servos.

    python -m cartesian_hand.sim --task zero
    python -m cartesian_hand.sim --task cap

Same task and direct-policy files as `studio.live`, not ports of them. Existing
tasks yield `Motions` programs; a direct policy instead receives the canonical
typed observation and returns goal, speed, and effort every tick. What differs
between here and hardware is the adapter around either controller -- `mj_step`
and a `qpos` read where the other has `set_positions` and `read_all`.

Grip force
----------
**Effort is a force cap, as on the servo.** `Motions` torque (0-1000 servo
units) and direct-policy `Action.effort_limit` (0-1) both reach `drive`, which
normalises them to 0-1 and hands them to the backend with the goal. The CPU
backend writes that into `actuator_forcerange` every tick, scaled by the
model's own full-scale force (the MJCF's +/-87 N), so `effort_limit = 0.08`
caps a joint at 7 N. A grip now yields at a tuned force and a `stop="stuck"`
can fire against an object rather than only against a joint limit.

The scale is the weak link, not the mechanism: the 87 N is inherited from the
sibling hand and was never measured for this one (see the `_force_note` in
`assets/cartesian_hand/source/kinematics.json`). Torques tuned on the bench
transfer as *ratios* until that is re-derived.

`run_warp` still drops effort: `mujoco_warp` keeps `actuator_forcerange` on
the model, shared by every world, and a per-world cap has not been wired or
measured there.

Two backends, one loop
----------------------
`run` is CPU MuJoCo at N=1 -- the reference, and the fallback on a box with no
CUDA. `run_warp` is `mujoco_warp` at any N. Neither is a port of the other:
both hand `drive` the same four verbs and it owns the loop, so there is exactly
one place where a tick is decided.

What makes the batched one worth having is that `Motions` is already torch, and
`wp.to_torch` views Warp's `qpos`/`ctrl` with no copy, so the whole control loop
stays resident on the GPU -- measured millimetres in and goals out never touch
the host. Measured on an RTX 4090 with the shipped model: 74.6k steps/s on CPU
at N=1 against 8.1M steps/s at N=4096, a 109x speedup. That figure is an upper
bound: the model has no objects in it yet, so `ncon` is 0 and collision is doing
no work. Re-measure once there is something to grasp.

**`mj_step`, not `mj_forward`.** The opposite of `studio.live`, and for the
opposite reason: there the physics *is* the hand and re-simulating would
overwrite what it reported, here the physics is all there is. Stepping is also
what applies the `<equality>` couplings, so a rack pair's follower moves on its
own rather than needing the explicit write the studio loop does.
"""
import dataclasses
import time

import mujoco
import numpy as np
import torch
import tyro
from collections.abc import Callable

from .config import DEFAULT_HAND, HandConfig, get_hand
from .mjcf import MM_PER_M, mjcf_path, narrow_ctrlrange, qpos_addrs
from .motions import Task, TaskRunner
from .policy import Policy, PolicyRunner, PolicyState
from . import tasks

SERVO_TORQUE_FULL_SCALE = 1000.0
"""The servo `torque` register's full scale: `Motions` torque / this is the
same 0-1 effort a direct policy's `Action.effort_limit` already is."""


def load(xml: str | None, cfg: HandConfig
         ) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """The model ready to drive, its data, and each actuator's full-scale force.

    A model with a keyframe starts from its first one: qpos 0 is the closed
    jaw, so a scene with an object in it (`objects.write_scene`) opens the jaws
    there rather than starting them inside the bottle.

    The full-scale force is read before anything caps it: it is what effort 1
    means, and `cap_force` overwrites the range it came from every tick.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path(xml))
    narrow_ctrlrange(model, cfg)
    full_force_n = np.abs(model.actuator_forcerange).max(axis=1)   # [nu]
    model.actuator_forcelimited[:] = 1
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
    return model, data, full_force_n


def cap_force(model: mujoco.MjModel, full_force_n: np.ndarray,
              effort: np.ndarray) -> None:
    """Cap each actuator at `effort` (0-1, [nu]) of its full-scale force."""
    cap_n = effort * full_force_n
    model.actuator_forcerange[:, 0] = -cap_n
    model.actuator_forcerange[:, 1] = cap_n


def step_limit_mm(cfg: HandConfig,
                  device: str | torch.device = "cpu") -> torch.Tensor:
    """`[1, J]` millimetres a joint may be commanded to advance in one tick.

    The servo profiles internally: `set_positions(..., speed=300)` means the
    joint tracks a setpoint that walks toward the goal at 300 counts/s, not that
    it appears there. mujoco's `<position>` actuators model none of that -- they
    are handed a target and the solver drives to it as hard as `kp` allows -- so
    without this a sim joint crosses 55 mm in the same tick it crosses 5.

    That is not a cosmetic difference, it is the one that made time meaningless
    in sim: zeroing finished in 45 ticks here against roughly 4000 on the bench,
    every distance and every timeout cost the same, and a search over travel and
    budgets found a completely flat objective (measured 2026-09-03). Stall
    detection is affected too -- `STUCK_WINDOW_STEPS` is 10 ticks of *travel*,
    and a joint that arrives instantly never spends them.
    """
    return (cfg.gain_vector("speed").to(device=device, dtype=torch.float32)
            / cfg.counts_per_mm / cfg.control_hz)[None, :]


def profile(goal_mm: torch.Tensor, command_mm: torch.Tensor,
            limit_mm: torch.Tensor) -> torch.Tensor:
    """Walk the *commanded* setpoint toward `goal_mm` at no more than `limit_mm`/tick.

    The setpoint is what the servo ramps, and the joint follows it with whatever
    error the load imposes. Ramping from the measured position instead would
    model a joint that cannot be pushed off course -- the opposite of a stall --
    and, more importantly, it can never let the setpoint outrun the joint by
    more than one step, which is exactly what `acc` does on a real servo and
    what is needed for a gravity-loaded DOF to develop enough force to lift.
    z could not rise against gravity here before this fix: against the bench it
    took `acc=200` to free the z stage.

    The first call seeds the setpoint with the joint's measured position, so a
    sim run starts from where it actually is.
    """
    return command_mm + (goal_mm - command_mm).clamp(-limit_mm, limit_mm)


def drive(controller: Task | Policy, cfg: HandConfig,
          read_mm: Callable[[], torch.Tensor],
          write_goal: Callable[[torch.Tensor, torch.Tensor], None],
          advance: Callable[[], None],
          sense: Callable[[], torch.Tensor | None],
          report: Callable[[], None],
          limit_mm: torch.Tensor, hold_torque: torch.Tensor,
          max_seconds: float,
          episode_over: Callable[[], bool] = lambda: False
          ) -> torch.Tensor | PolicyState | None:
    """The control loop both backends run, and the only copy of it.

    A backend supplies four verbs -- read millimetres, write a goal and its
    effort, advance the physics, sample an external contact signal -- and
    differs in nothing else. Effort reaches `write_goal` normalised to 0-1
    whichever controller produced it, so a backend never sees servo units.
    Keeping the loop here is what stops a fix landing in one backend only.

    `report` runs after the loop and *before* the failure check, so a run that
    ends badly still prints where the hand got to.

    A legacy generator returns its measurement; a direct policy its typed final
    state. Raises `RuntimeError` if any env failed -- a probe that closed on
    air, a DOF that never found its stop. Tasks report that as `Result.ok`
    rather than raising (see `motions.Result`); turning it back into an
    exception is this backend's policy, not the task's, since a CLI that printed
    a radius after measuring nothing is worse than one that stops. A caller
    wanting the flags per env drives `TaskRunner` and reads `runner.result`.

    `episode_over` ends the run early, checked after every tick, and returns
    None: the scene, not the controller, decided it was over (a cap lifted
    off, say), so the controller has no result to report and no failure to
    raise.
    """
    hz = cfg.control_hz
    direct = (PolicyRunner(controller, hz)
              if isinstance(controller, Policy) else None)
    runner = (None if direct is not None else
              TaskRunner(controller, hold_torque=hold_torque))

    # The setpoint lives across ticks: a fresh one per tick would never outrun
    # the joint by more than one step, and a gravity-loaded DOF would not
    # develop enough force to lift. Seeded from the first reading so the run
    # starts from where it actually is.
    setpoint: torch.Tensor | None = None
    for _ in range(int(max_seconds * hz)):
        mm = read_mm()
        if setpoint is None:
            setpoint = mm.clone()
        external = sense()
        if direct is not None:
            if direct.finished():
                break
            contact = (torch.as_tensor(external, dtype=torch.bool,
                                       device=mm.device)
                       if external is not None else None)
            action = direct.tick(mm, contact)
            goal, tick_limit = action.goal_mm, action.max_speed_mm_s / hz
            effort = action.effort_limit
        else:
            step = runner.tick(mm, external)
            if step is None:
                break
            goal, tick_limit = step[0], limit_mm
            effort = step[1] / SERVO_TORQUE_FULL_SCALE
        # mujoco clamps ctrl to the range `narrow_ctrlrange` set, which is what
        # turns zeroing's deliberate 120mm overtravel into "drive to the closed
        # end and lean on it" rather than an unreachable goal.
        setpoint = profile(goal, setpoint, tick_limit)
        write_goal(setpoint, effort.clamp(0.0, 1.0))
        advance()
        if episode_over():
            report()
            return None
    else:
        raise RuntimeError(
            f"task did not finish within {max_seconds}s of simulated time")

    report()
    if direct is not None:
        if direct.failed():
            raise RuntimeError("direct policy failed")
        return direct.state
    value, ok, why = runner.result
    if not bool(ok.all()):
        raise RuntimeError(why)
    return value


def run(controller: Task | Policy, cfg: HandConfig, xml: str | None = None,
        swap: bool = False,
        max_seconds: float = 180.0, verbose: bool = True,
        external_signal: Callable[[mujoco.MjModel, mujoco.MjData],
                                  torch.Tensor | None] | None = None,
        viewer: bool = False,
        episode_over: Callable[[], bool] = lambda: False
        ) -> torch.Tensor | PolicyState | None:
    """CPU MuJoCo at N=1: the reference backend, and the one with no CUDA.

    `max_seconds` is simulated time and is a backstop, not a schedule: a task
    that ends on its own stop conditions finishes long before it, and one that
    does not is a bug worth stopping rather than a run worth waiting out.

    `viewer` opens MuJoCo's passive viewer and paces the run to real time.
    `episode_over` is `drive`'s.

    Effort caps each actuator's force every tick -- see "Grip force" in the
    module docstring.
    """
    model, data, full_force_n = load(xml, cfg)
    view = None
    if viewer:
        from mujoco import viewer as mj_viewer   # `import mujoco.viewer` here
        view = mj_viewer.launch_passive(model, data)  # would shadow `mujoco`

    # One joint per DOF is enough to read: a rack pair's two joints are held
    # equal by the model's own constraint, so the follower carries no
    # information the leader does not.
    lead = np.array([row[0] for row in qpos_addrs(model, swap)])
    substeps = max(1, round((1.0 / cfg.control_hz) / model.opt.timestep))

    def advance() -> None:
        tick = time.perf_counter()
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        if view is not None:
            view.sync()
            time.sleep(max(0.0, 1.0 / cfg.control_hz
                           - (time.perf_counter() - tick)))

    def write_goal(goal_mm: torch.Tensor, effort: torch.Tensor) -> None:
        data.ctrl[:] = goal_mm[0].numpy() / MM_PER_M
        cap_force(model, full_force_n, effort[0].numpy())

    def report() -> None:
        if verbose:
            pos = " ".join(f"{v:6.1f}" for v in data.qpos[lead] * MM_PER_M)
            print(f"[sim] finished at mm [{pos} ]")

    try:
        return drive(
            controller, cfg,
            read_mm=lambda: torch.from_numpy(
                data.qpos[lead] * MM_PER_M).float()[None, :],
            write_goal=write_goal,
            advance=advance,
            sense=lambda: external_signal(model, data) if external_signal else None,
            report=report,
            limit_mm=step_limit_mm(cfg),
            hold_torque=cfg.gain_vector("torque_min_to_move")
            .to(torch.float32)[None, :],
            max_seconds=max_seconds, episode_over=episode_over)
    finally:
        if view is not None:
            view.close()


def run_warp(controller: Task | Policy, cfg: HandConfig, xml: str | None = None,
             swap: bool = False, max_seconds: float = 180.0,
             verbose: bool = True, n_envs: int = 1,
             device: str | torch.device = "cuda",
             external_signal: Callable[[object, object],
                                       torch.Tensor | None] | None = None
             ) -> torch.Tensor | PolicyState:
    """`run`, batched over `n_envs` on the GPU via `mujoco_warp`.

    Same contract as `run` and the same loop: read millimetres, tick the task,
    write goals, advance the physics. The differences are all in those three
    verbs -- `wp.to_torch` views instead of a numpy read, and a captured Warp
    graph instead of a Python `mj_step` loop.

    `qpos` and `ctrl` are zero-copy torch views of Warp's own buffers, so
    assigning into `ctrl` IS the write to the simulator. The task's tensors were
    built on `start_mm.device`, which `main` puts on the GPU, so nothing in the
    tick synchronises with the host.

    The substep block is captured once as a Warp graph rather than launched as
    `substeps` separate Python calls: at 50 Hz control against a 0.5 ms timestep
    that is 40 launches per tick, and the launch overhead is the loop's cost
    once the physics itself is batched.

    Effort is dropped here, unlike in `run` -- see "Grip force" in the module
    docstring.
    """
    import mujoco_warp as mjw
    import warp as wp

    model = mujoco.MjModel.from_xml_path(mjcf_path(xml))
    narrow_ctrlrange(model, cfg)                  # before put_model: it copies
    lead = [row[0] for row in qpos_addrs(model, swap)]

    m = mjw.put_model(model)
    d = mjw.make_data(model, nworld=n_envs)
    qpos, ctrl = wp.to_torch(d.qpos), wp.to_torch(d.ctrl)   # [N, nq], [N, nu]
    substeps = max(1, round((1.0 / cfg.control_hz) / model.opt.timestep))

    # Warm up to compile the kernels, then rewind: that first step advanced the
    # sim, and the task's first measurement must be the rest pose.
    mjw.step(m, d)
    wp.synchronize()
    mjw.reset_data(m, d)
    with wp.ScopedCapture() as capture:
        for _ in range(substeps):
            mjw.step(m, d)
    graph = capture.graph

    def write_goal(goal_mm: torch.Tensor, effort: torch.Tensor) -> None:
        ctrl[:] = goal_mm / MM_PER_M

    def report() -> None:
        if verbose:
            pos = " ".join(f"{v:6.1f}"
                           for v in (qpos[0, lead] * MM_PER_M).tolist())
            print(f"[sim] {n_envs} env(s) finished, env 0 at mm [{pos} ]")

    return drive(
        controller, cfg,
        read_mm=lambda: qpos[:, lead] * MM_PER_M,
        write_goal=write_goal,
        advance=lambda: wp.capture_launch(graph),
        sense=lambda: external_signal(m, d) if external_signal else None,
        report=report,
        limit_mm=step_limit_mm(cfg, device),
        hold_torque=cfg.gain_vector("torque_min_to_move")
        .to(device=device, dtype=torch.float32)[None, :],
        max_seconds=max_seconds)


def override(task_cfg: object | None, pairs: tuple[str, ...],
             task: str) -> object:
    """`task_cfg` with each `FIELD=VALUE` in `pairs` applied, typed like the
    field it replaces. Raises naming the fields there are."""
    if task_cfg is None:
        raise ValueError(f"task {task!r} has no Config to --set")
    fields = {f.name: f for f in dataclasses.fields(task_cfg)}
    changes = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or name not in fields:
            raise ValueError(f"--set {pair!r}: want FIELD=VALUE, FIELD one of "
                             f"{sorted(fields)}")
        kind = type(getattr(task_cfg, name))
        changes[name] = (value.lower() in ("1", "true", "yes") if kind is bool
                         else kind(value))
        print(f"[sim] {task}.{name} = {changes[name]!r}")
    return dataclasses.replace(task_cfg, **changes)


def main(task: str = "zero",
         hand: str = DEFAULT_HAND,
         xml: str | None = None,
         swap: bool = False,
         max_seconds: float = 180.0,
         n_envs: int = 1,
         warp: bool = False,
         obj: str | None = None,
         viewer: bool = False,
         set: tuple[str, ...] = (),
         full_cycle: bool = False) -> None:
    """Run one task in simulation.

    Args:
        task: a file stem under `tasks/`, e.g. `zero` or `cap`. An unknown name
            raises with the registry listed.
        hand: which entry in `config.HANDS` supplies travel and control rate.
        xml: MJCF path. Default $CARTESIAN_HAND_MJCF, then this repo's bundled
            `assets/cartesian_hand/cartesian_hand.xml`.
        swap: exchange finger DOFs 1<->2 and 5<->6.
        max_seconds: simulated-time backstop.
        n_envs: how many worlds to simulate. Above 1 implies --warp.
        warp: use the GPU `mujoco_warp` backend instead of CPU mujoco.
        obj: an object from assets/objects/bottles.json to put in the jaws,
            e.g. `water_500ml`. Builds its scene (overrides --xml), sets the
            task's `cap_offset` from it, and reports whether the cap came off.
            CPU only.
        viewer: watch the run in MuJoCo's viewer, paced to real time. CPU only.
        set: override fields of the task's `Config`, e.g. `--set
            squeeze_torque=20 lift_torque=300`. Applied after --obj, so it
            wins. Torques are servo units, 0-1000.
        full_cycle: with --obj, keep running after the cap is lifted, through
            the closing half the scene does not model. By default the run ends
            the tick the cap is lifted off.
    """
    cfg = get_hand(hand)
    warp = warp or n_envs > 1
    # `start_mm` fixes the batch width AND the device for every tensor the task
    # builds, since `Program` is constructed with `start_mm.device`. Putting it
    # on the GPU is what keeps the whole tick off the host.
    device = "cuda" if warp else "cpu"
    start = torch.zeros(n_envs, cfg.n_dof, device=device)
    monitor, kwargs = None, {}
    if obj is not None:
        if warp:
            raise ValueError("--obj runs on the CPU backend only")
        from . import objects
        bottle = objects.load(obj)
        xml = objects.write_scene(bottle)
        monitor = objects.ThreadMonitor(bottle)
        task_cfg = objects.teacher_config(tasks.config(task), bottle)
        if task_cfg is not None:
            kwargs["cfg"] = task_cfg
    if set:
        kwargs["cfg"] = override(kwargs.get("cfg") or tasks.config(task),
                                 set, task)
    controller = tasks.make(task, cfg, start, **kwargs)
    try:
        result = (run_warp(controller, cfg, xml=xml, swap=swap,
                           max_seconds=max_seconds,
                           n_envs=n_envs, device=device) if warp else
                  run(controller, cfg, xml=xml, swap=swap,
                      max_seconds=max_seconds, external_signal=monitor,
                      viewer=viewer,
                      episode_over=(lambda: monitor.opened)
                      if monitor is not None and not full_cycle
                      else lambda: False))
    finally:
        if monitor is not None:
            print(monitor.summary())
    if result is None:
        print(f"[sim] {task}: ended when the cap was lifted (--full-cycle "
              f"to run the closing half too)")
    elif isinstance(controller, Policy):
        print(f"[sim] {task}: done={result.done.tolist()}, "
              f"failed={result.failed.tolist()}")
    else:
        print(f"[sim] {task}: {[round(float(v), 2) for v in result[0].flatten()]}"
              if result.ndim > 1 else f"[sim] {task}: {result.tolist()}")


if __name__ == "__main__":
    tyro.cli(main, prog="python -m cartesian_hand.sim")
