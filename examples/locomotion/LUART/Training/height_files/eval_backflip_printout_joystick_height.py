import argparse
import copy
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import pickle

import numpy as np
import torch
import pygame  # For joystick input
from reward_wrapper_height import Go2
from rsl_rl.runners import OnPolicyRunner


import genesis as gs

# Global variable to track desired body height
body_height = 0.2  # Initial desired body height (range: 0.1 to 0.35)

def export_policy_as_jit(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{name}.pt')
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

def get_joystick_input():
    """
    Reads joystick input and maps it to desired values for control inputs.
    Returns a tuple (A_x, A_y, ang_vel) based on joystick axis positions.
    """
    pygame.event.pump()  # Process joystick events
    # (Note: Adjust axis indices and scaling as needed.)
    x_axis = joystick.get_axis(1)       # Axis 1: linear velocity x
    y_axis = joystick.get_axis(0)       # Axis 0: linear velocity y
    angular_axis = joystick.get_axis(3) # Axis 3: angular velocity

    # Map joystick values to control input ranges
    A_x = 1.0 * x_axis
    A_y = 1.0 * y_axis
    ang_vel = 1.0 * angular_axis
    return A_x, A_y, ang_vel

def update_body_height():
    """
    Adjusts the desired body height based on button presses.
    Uses button X (index 1) to increase and button B (index 3) to decrease the height.
    """
    global body_height
    increment = 0.02  # Height adjustment step
    pygame.event.pump()  # Process joystick events
    if joystick.get_button(1):  # Button X increases height
        body_height = min(body_height + increment, 0.35)
    if joystick.get_button(3):  # Button B decreases height
        body_height = max(body_height - increment, 0.1)

def print_labeled_obs(obs):
    """
    Prints the observation tensor with labeled components.
    (You might need to adjust the slicing if your observation space has changed.)
    """
    # ang_vel = obs[0:3]
    # projected_gravity = obs[3:6]
    # # Updated slicing to include the height command (assuming command vector now has 4 elements)
    # commands = obs[6:10]
    # dof_pos = obs[10:18]
    # dof_vel = obs[18:26]
    # actions = obs[26:34]
    height11 = obs[34]
    commands = obs[9]

    # print("Observations:")
    # print(f"  Angular Velocity (ang_vel): {ang_vel}")
    # print(f"  Projected Gravity: {projected_gravity}")
    # print(f"  Commands: {commands}")
    # print(f"  DOF Positions (dof_pos): {dof_pos}")
    # print(f"  DOF Velocities (dof_vel): {dof_vel}")
    # print(f"  Actions: {actions}")
    print(f"  Height: {height11}")
    print(f"  Command Height: {commands}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='newtt-new-tracking-height')
    parser.add_argument('-v', '--vis', action='store_true', default=True)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=700)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    # Initialize joystick
    pygame.init()
    pygame.joystick.init()
    global joystick
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    else:
        raise RuntimeError("No joystick connected!")

    # Load environment and training configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(
        open(f'logs/{args.exp_name}/cfgs.pkl', 'rb')
    )
    # Optionally modify configs as needed.
    env_cfg["episode_length_s"] = 1000.0  # Set the episode length

    env = Go2(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=True,
        debug=True,
    )

    log_dir = f'logs/{args.exp_name}'
    jit_ckpt_path = os.path.join(log_dir, 'exported', args.exp_name + f'_ckpt{args.ckpt}.pt')
    if os.path.exists(jit_ckpt_path):
        policy = torch.jit.load(jit_ckpt_path)
        policy.to(device='cuda:0')
    else:
        # If no exported JIT exists, load from checkpoint and export the actor.
        args.max_iterations = 1
        from train_walk_tt_height_student_knows import get_train_cfg
        runner = OnPolicyRunner(env, get_train_cfg(args), log_dir, device='cuda:0')
        resume_path = os.path.join(log_dir, f'model_{args.ckpt}.pt')
        runner.load(resume_path)
        export_path = os.path.join(log_dir, 'exported')
        export_policy_as_jit(runner.alg.actor_critic, export_path, args.exp_name + f'_ckpt{args.ckpt}')
            
        policy = runner.get_inference_policy(device='cuda:0')

    # Reset the environment to get the initial observation.
    obs, _ = env.reset()

    # Optionally start recording if enabled.
    if args.record:
        env.start_recording(record_internal=False)



    with torch.no_grad():
        while True:
            # Get joystick input and update the environment's command signal.
            A_x, A_y, ang_vel = get_joystick_input()
            update_body_height()  # Adjust desired body height based on button presses

            env.commands[:, 0] = A_x      # Linear velocity x
            env.commands[:, 1] = A_y      # Linear velocity y
            env.commands[:, 2] = ang_vel  # Angular velocity
            env.commands[:, 3] = body_height  # Desired body height

            # Compute the next action based on the updated observations.
            obs, extras = env.get_observations()
            actions = policy(obs)
            # obs, _, rews, dones, infos = env.step(actions)
            obs, rews, dones, infos = env.step(actions)

            # Print out the labeled observation for debugging.
            print_labeled_obs(obs[0])  # Assuming a single environment

            # Reset the environment if the episode ends.
            if dones[0]:
                obs, _ = env.reset()

if __name__ == '__main__':
    main()
