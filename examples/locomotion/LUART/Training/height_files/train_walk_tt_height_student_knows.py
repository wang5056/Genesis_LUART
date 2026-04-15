import argparse
import copy
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import pickle
import shutil

import numpy as np
import torch
# import wandb
from reward_wrapper_height import Go2
from locomotion_env_height_self_study import LocoEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(args):
    train_cfg_dict = {
        'algorithm': {
            'class_name': 'PPO',  # Already added
            'clip_param': 0.2,
            'desired_kl': 0.01,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': 0.001,
            'max_grad_norm': 1.0,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'schedule': 'adaptive',
            'use_clipped_value_loss': True,
            'value_loss_coef': 1.0,
        },
        'init_member_classes': {},
        'policy': {
            'class_name': 'ActorCritic',  # Already added
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0,
        },
        'runner': {
            'algorithm_class_name': 'PPO',
            'checkpoint': -1,
            'experiment_name': args.exp_name,
            'load_run': -1,
            'log_interval': 1,
            'max_iterations': args.max_iterations,
            'num_steps_per_env': 24,  # Keep for consistency
            'policy_class_name': 'ActorCritic',
            'record_interval': 50,
            'resume': False,
            'resume_path': None,
            'run_name': '',
            'runner_class_name': 'runner_class_name',
            'save_interval': 100,
        },
        'runner_class_name': 'OnPolicyRunner',
        'num_steps_per_env': 24,  # Add this line at top level
        'empirical_normalization': False,  # Add default value (see below)
        'save_interval': 100,  # Add default value (see below)
        'seed': 1,
    }
    return train_cfg_dict

# /home/tt/Genesis/genesis/assets/urdf/turtle_test/urdf/turtle1_tt.urdf
def get_cfgs():
    env_cfg = {
        'urdf_path': '/home/tt/Genesis/genesis/assets/urdf/ttt_chassis/urdf/ttt_chassis.urdf',
        'links_to_keep': ['fl_foot', 'fr_foot', 'rl_foot', 'rr_foot','chassis','fl_k', 'rl_k','rr_k','fr_k'],
        'num_actions': 8,
        'num_dofs': 8,
        'default_joint_angles': {  # [rad]
            'fl_a_joint': 1.3,
            'rl_a_joint': -1.3, 
            'rr_a_joint': 1.3,
            'fr_a_joint': -1.3,

            'fl_h_joint': 0.15,
            'rl_h_joint': -0.15,
            'rr_h_joint': 0.15,
            'fr_h_joint': -0.15,
        },
        'dof_names': [
            'fl_a_joint',
            'rl_a_joint',
            'rr_a_joint',
            'fr_a_joint',
            'fl_h_joint',
            'rl_h_joint',
            'rr_h_joint',
            'fr_h_joint',
        ],
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'fl_k', 'rl_k','rr_k','fr_k'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],


        # PD
        'PD_stiffness': {'joint': 30.0},
        'PD_damping': {'joint': 1.5},
        'use_implicit_controller': False,


        # termination
        'termination_if_roll_greater_than': 0.3,
        'termination_if_pitch_greater_than': 0.3,
        'termination_if_height_lower_than': 0.005,


        # base pose
        'base_init_pos': [0.0, 0.0, 0.33],
        'base_init_quat': [1.0, 0.0, 0.0, 0.0],

        
        # random push
        'push_interval_s': -1,
        'max_push_vel_xy': 1.0,


        # time (second)
        'episode_length_s': 20.0,
        'resampling_time_s': 4.0,
        'command_type': 'ang_vel_yaw',  # 'ang_vel_yaw' or 'heading'
        'action_scale': 0.25,
        'action_latency': 0.04,
        'clip_actions': 100.0,
        'send_timeouts': True,
        'control_freq': 25,
        'decimation': 4,
        'feet_geom_offset': 1,
        'use_terrain': False,


        # domain randomization
        'randomize_link_length': True,
        'link_length_range': [0.6, 1.1],
        'link_names_to_randomize': ['fl_k', 'rl_k', 'rr_k', 'fr_k'],
        'randomize_friction': True,
        'friction_range': [0.2, 1.5],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 2.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.8, 1.2],
        
        # coupling
        'coupling': False,


    }
    obs_cfg = {
        'num_obs': 9 + 2 + 3 * env_cfg['num_dofs'], # ths + 2 for commands_height and height sensor. 
        'num_history_obs': 1,
        'obs_noise': {
            'ang_vel': 0.1,
            'gravity': 0.02,
            'dof_pos': 0.01,
            'dof_vel': 0.5,
            'height_info':0.02,
        },
        'obs_scales': {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
            'height_info':1.0,
        },
        'num_priv_obs': 12 + 2 + 4 * env_cfg['num_dofs'], # the +1 is for commands
    }
    reward_cfg = {
        'tracking_sigma': 0.25,
        'soft_dof_pos_limit': 0.9,
        'base_height_target': 0.2,
        # 'reward_scales': {
        #     'tracking_lin_vel': 1.0,
        #     'tracking_ang_vel': 0.5,
        #     'lin_vel_z': -2.0,
        #     'ang_vel_xy': -0.05,
        #     'orientation': -10.,
        #     'base_height': -40.,  # default: -50
        #     'torques': -0.0002,
        #     'collision': -1.,
        #     'dof_vel': -0.,
        #     'dof_acc': -2.5e-7,
        #     'feet_air_time': 1.0,  #1.0
        #     'collision': -1.,
        #     'action_rate': -0.01,
        # },
        'reward_scales': {
            'tracking_lin_vel': 1.0,      # Increase from 1.0 to 2.5 for stronger emphasis
            'tracking_ang_vel': 0.5,      # Unchanged
            'lin_vel_z': -0.5,            # Reduce from -2.0 to -0.5 to allow vertical motion
            'ang_vel_xy': -0.05,          # Unchanged
            'orientation': -5.0,          # Reduce from -10.0 to -5.0 for more flexibility
            'base_height': -30.0,         # Reduce from -40.0 to -10.0 to allow height variation
            'torques': -0.00008,           # Unchanged   -0.0002,
            'collision': -1.0,            # Unchanged
            'dof_vel': -0.0,              # Unchanged
            'dof_acc': -2.5e-7,           # Unchanged
            'feet_air_time': 0.2,         # Increase from 1.0 to 2.5 to reward high leg lifts
            'action_rate': -0.01,         # Unchanged
        },
    }
    command_cfg = {
        'num_commands': 5,
        'lin_vel_x_range': [-1.0, 1.0],
        'lin_vel_y_range': [-1.0, 1.0],
        'ang_vel_range': [-1.0, 1.0],
        'height_range': [0.03, 0.32],

    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='newtt-knowing-the-height-only-height-change-wider')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=12000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    log_dir = f'logs/{args.exp_name}'
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
    )

    runner = OnPolicyRunner(env, get_train_cfg(args), log_dir, device='cuda:0')

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    # wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg],
        open(f'{log_dir}/cfgs.pkl', 'wb'),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''