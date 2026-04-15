import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import pickle
import pygame  # For joystick input
import torch
from turtle_env_height import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# Global variable to track desired body height
body_height = 0.2  # Initial desired body height, within the range [0.1, 0.35]

def get_joystick_input():
    """
    Reads joystick input and maps it to desired values for control inputs.
    Returns a tuple (A_x, A_y, ang_vel) based on joystick axis positions.
    """
    pygame.event.pump()  # Process joystick events
    x_axis = joystick.get_axis(0)  # Axis 0: linear velocity x
    y_axis = joystick.get_axis(1)  # Axis 1: linear velocity y
    angular_axis = joystick.get_axis(3)  # Axis 3: angular velocity

    # Map joystick values to control input ranges
    A_x = 0.8 * y_axis  # Map x-axis to [-0.8, 0.8]
    A_y = 0.6 * x_axis  # Map y-axis to [-0.6, 0.6]
    ang_vel = 1 * angular_axis  # Map angular velocity to [-1, 1]
    return A_x, A_y, ang_vel

def update_body_height():
    """
    Adjusts the desired body height based on button presses.
    Uses button X (1) to increase and button B (2) to decrease the height.
    """
    global body_height
    increment = 0.01  # Increment step for height adjustment
    pygame.event.pump()  # Process joystick events
    if joystick.get_button(1):  # Button X
        body_height = min(body_height + increment, 0.35)  # Cap at max height
    if joystick.get_button(3):  # Button B
        body_height = max(body_height - increment, 0.1)  # Cap at min height

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="turtle3-walking-fastturn15")
    parser.add_argument("--ckpt", type=int, default=800)
    args = parser.parse_args()

    gs.init()

    # Initialize joystick
    pygame.init()
    pygame.joystick.init()
    global joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["episode_length_s"] = 100.0  # Set this to the desired length, longer lead to uninterrupted simulations

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    with torch.no_grad():
        while True:
            # Get joystick input
            A_x, A_y, ang_vel = get_joystick_input()
            update_body_height()  # Adjust body height based on button presses

            # Update commands directly in the environment
            env.commands[:, 0] = A_x  # Linear velocity x
            env.commands[:, 1] = A_y  # Linear velocity y
            env.commands[:, 2] = ang_vel  # Angular velocity
            env.commands[:, 3] = body_height  # Desired body height

            # Perform simulation step
            actions = policy(env.get_observations())  # Use updated observations with joystick commands
            obs, _, rews, dones, infos = env.step(actions)

            # Reset environment if needed
            if dones[0]:  # Assuming single environment
                obs, _ = env.reset()

if __name__ == "__main__":
    main()
