# Genesis_LUART — Height-Aware Quadruped Locomotion

PPO training and joystick evaluation for the LUART quadruped robot in the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator. Adds a
**commanded base-height** to the policy on top of standard locomotion
commands (lin-vel xy, ang-vel yaw), with several student-policy variants
(privileged-height, self-study, terrain-aware).

## Repository layout

```
.
├── train_walk_tt_height_student_knows.py        # student receives height command
├── train_walk_tt_height_student_dont_know.py    # student must infer height (self-study)
├── train_walk_tt_height_student_knows_liveshow.py
├── train_walk_tt_height_diff_terrain.py         # height training over varied terrain
├── eval_backflip_printout_joystick_height.py    # joystick eval (all variants)
│
├── reward_wrapper_height.py                     # Go2 env wrapper (rewards)
├── reward_wrapper_height_dont_know.py
│
├── locomotion_env_height.py                     # baseline height-aware env
├── locomotion_env_height_student_knows.py       # height fed into actor obs
├── locomotion_env_height_self_study.py          # height fed only into critic
├── locomotion_env_height_student_knows_liveshow.py
├── locomotion_env_height_sand.py                # MPM sand terrain variant
│
└── utils.py                                     # math + helpers
```

## Setup

```bash
# 1. Genesis simulator (follow upstream install)
pip install genesis-world

# 2. RL trainer
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl && pip install -e . && cd ..

# 3. Misc deps
pip install torch numpy pygame
```

The training scripts also expect a URDF asset at:

```
/home/tt/Genesis/genesis/assets/urdf/ttt_chassis/urdf/ttt_chassis.urdf
```

Edit `urdf_path` inside `get_cfgs()` in each training script to point at
your local copy.

## Training

All training commands take an experiment name (`-e`) used as the
`logs/<name>/` directory. The script **deletes** `logs/<name>/` if it
already exists, so use a fresh name per run.

```bash
# student receives the commanded body height (privileged)
python train_walk_tt_height_student_knows.py -e my_run

# student infers body height from proprioception (self-study)
python train_walk_tt_height_student_dont_know.py -e my_run_selfstudy

# height training on varied terrain
python train_walk_tt_height_diff_terrain.py -e my_run_terrain
```

Common flags: `-B <num_envs>` (default 12000), `--max_iterations` (default
1000), `-c` CPU only, `-v` show viewer, `--resume <prev_exp> --ckpt N` to
warm-start.

Checkpoints land in `logs/<name>/model_<iter>.pt` every 100 iterations.

## Evaluation (joystick)

A USB joystick is required.

```bash
python eval_backflip_printout_joystick_height.py -e <exp_name> --ckpt <iter>
```

Controls:
- Left stick → linear velocity x/y
- Right stick X → angular velocity (yaw)
- Button 1 → raise body height (+0.02 m, max 0.35)
- Button 3 → lower body height (-0.02 m, min 0.10)

## Notes

- All training scripts rely on `from utils import *` and import the
  matching `locomotion_env_*` and `reward_wrapper_height*` modules from
  the same directory. Run scripts from this directory so the relative
  imports resolve.
- The default `num_envs=12000` requires a GPU with ~10 GB of free VRAM.
  Reduce with `-B` if you OOM.
