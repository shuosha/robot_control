import os
from pathlib import Path
import argparse
import multiprocess as mp
import sys

from robot_control.utils.utils import get_root
root = get_root(__file__)
sys.path.append(str(root / "logs"))

import torch
import json
import numpy as np
from robot_control.modules.robot_env import RobotEnv


"""
Example usage:  
    python scripts/teleop.py replay data.npz
"""

def load_npz(file_path: str, eps_idx: int):
    data = np.load(file_path, allow_pickle=True)
    eps_idx = f"episode_{eps_idx:04d}"
    cart_traj = np.concatenate((
        data[f'{eps_idx}/action.eef_pos'], 
        data[f'{eps_idx}/action.eef_quat'],
        data[f'{eps_idx}/action.gripper']
    ), axis=1)
    init_pose = np.concatenate((
        data[f'{eps_idx}/obs.qpos'][0] * 180.0 / np.pi,
        data[f'{eps_idx}/obs.gripper'][0]
    ))
    return cart_traj, init_pose

if __name__ == '__main__':
    # mp.set_start_method('spawn')  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('rpz_path', type=str, default='')
    parser.add_argument('--eps_idx', type=int, default=0)
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--pusht', action='store_true', default=False) # NOTE: not supported yet
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    parser.add_argument('--fps', type=float, default=10.0, help='camera recording and agent update fps')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"
    assert os.path.exists(args.rpz_path), f"Replay file {args.rpz_path} does not exist"

    cart_traj, init_pose = load_npz(args.rpz_path, args.eps_idx)
    assert len(init_pose) == 8, "xarm7 requires 8 DOF initial pose"

    env = RobotEnv(
        exp_name=args.name,
        data_dir="replay",
        debug=True,

        resolution=(848, 480),
        capture_fps=30,
        record_fps=30,
        perception_process_func=None,

        # robot
        robot_name="xarm7",
        robot_ip=[args.robot_ip],
        bimanual=args.bimanual,
        gripper_enable=False if args.pusht else True,

        # control
        control_mode="position_control",
        admittance_control=True,
        ema_factor=1.0,
        action_agent_fps=10.0, # teleop & policy
        pusht_mode=args.pusht,
        action_receiver="replay",
        action_traj=cart_traj, # (ts, 8) pos + quat + gripper # degrees
        init_pose=init_pose, # (8,) # initial qpos + gripper
    )
    
    env.start()
    env.join()
