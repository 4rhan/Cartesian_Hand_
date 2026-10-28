"""Composed in the studio: 1 rows.

Written by `cartesian_hand.compose`. An ordinary task module: one
program, 1 steps, run in the order below. Every number is a
`Config` field with declared bounds, so the studio's tune panel can
move it, and `--task open_scisors` runs it on the hand or in mujoco.
"""
from dataclasses import dataclass, field

import torch

from ..config import HandConfig
from ..motions import Program, Result, Task, When


@dataclass
class Config:
    label: str = ""
    sets_datum: bool = False
    # base left finger+base right finger: goal from abs
    goal_0: float = 0.0
    torque_0: float = field(default=50.0, metadata={"tune": (25.0, 100.0)})


def build(hand: HandConfig, start_mm: torch.Tensor,
          cfg: Config | None = None, **kwargs) -> Task:
    cfg = cfg or Config()
    n_envs, n_dof = start_mm.shape
    p = Program(n_envs, n_dof, hand.control_hz, start_mm.device)
    p.step().set([1, 2], cfg.goal_0, cfg.torque_0,
                 'goal', 6.0, when=None, frame='abs')
    program = p.build()
    measured = yield program
    ok = (program.succeeded() | ~program.executed).all(dim=(1, 2))
    why = ("" if bool(ok.all()) else
           f"composed task failed in envs {(~ok).nonzero().flatten().tolist()}")
    return Result(program.stopped_at, ok, why)
