# Bench scripts

Unsupported one-off tools. They are not part of the control stack, nothing in
`cartesian_hand/` imports them, and their dependencies are deliberately left out
of `pyproject.toml`. Expect to edit them before they run.

| | |
|---|---|
| `ft_servo_tools/` | Servo bench CLI: `cli.py scan`, `set-id`, `gui`. Takes a port, knows nothing about hands. See `cartesian_hand/src/ft_servo/README.md`. |
| `bench_servo_check.py` | Drive one servo 200 counts from wherever it is parked. `PORT` and `SERVO_ID` are constants at the top; edit them first. |
| `sts3215_check.py` | Feetech STS3215 (not HLS): ping one servo, turn it a quarter turn and back. Pure pyserial, no build. `--scan` lists IDs, `--id` defaults to the only servo on the bus; `--port`/`--counts` override defaults. |
| `sts3215_zero.py` | **Run first.** Zero one STS3215: creep into its DOF's closed-end hard stop at the torque floor, write the servo's offset so the stop reads 4000, park at mid travel. The zero lives in the servo and survives power cycles. `--no-save` finds the stop without writing. |
| `sts3215_twin.py` | One STS3215 mirrored into the MuJoCo model: drag its joint's slider and the servo follows, `--teach` to turn the horn by hand and watch the joint. ID 2 is hand_3's DOF 2 by default. |
| `franka_arm_testing/` | Hand-on-arm experiments. Needs a Franka Panda and the `franky` package. `ee_prompt.py` is an interactive jog prompt, `sequence.py` a recorded pick-and-place. |
| `sync_sim_asset.py` | Re-copies the MuJoCo model into `assets/` from a `legged_env_v2` checkout, which is not public. The generated model is already bundled, so you do not need this. |
