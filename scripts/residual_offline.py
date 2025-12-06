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
    python scripts/teleop.py teleop_test
"""

if __name__ == '__main__':
    # cv2 encounter error when using multi-threading, use tk instead
    # cv2.setNumThreads(cv2.getNumberOfCPUs())
    # cv2.namedWindow("real env monitor", cv2.WINDOW_NORMAL)

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--pusht', action='store_true', default=False)
    parser.add_argument('--input_mode', type=str, default='residual_offline', choices=["gello", "keyboard", "residual", "residual_offline"])
    parser.add_argument('--init_pose', type=list, default=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0, 0.0])
    parser.add_argument('--robot', type=str, default='xarm7', choices=['xarm7', 'aloha', 'uf850'])
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    parser.add_argument('--fp_dir', default=None, type=str, help='foundation pose directory')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"
    if args.robot == "xarm7":
        assert len(args.init_pose) == 8, "xarm7 requires 8 DOF initial pose"
    elif args.robot == "uf850":
        assert len(args.init_pose) == 7, "uf850 requires 7 DOF initial pose"
    else:
        raise NotImplementedError(f"Robot {args.robot} not supported yet")

    init_poses = [np.array([0.048027075827121735,
        -0.13141998648643494,
        0.04114877060055733,
        0.65958571434021,
        0.09474538266658783,
        0.8325322866439819,
        -0.002133825793862343,
        0.0024999999441206455], dtype=np.float32)]  # teleop test pose
    init_poses[0][:7] *= 180.0 / np.pi  # rad to degrees
    print("Using initial pose: ", init_poses)

    env = RobotEnv(
        exp_name=args.name,
        data_dir="residual_offline",
        debug=True,

        resolution=(848, 480),
        capture_fps=30,
        record_fps=15,
        perception_process_func=None,
        foundation_pose_dir=args.fp_dir,

        # robot
        robot_name=args.robot,
        robot_ip=[args.robot_ip],
        bimanual=args.bimanual,
        gripper_enable=False if args.pusht else True,

        # control
        control_mode="position_control",
        admittance_control=True,
        ema_factor=1.0,
        action_agent_fps=15.0, # teleop & policy
        pusht_mode=args.pusht,
        action_receiver=args.input_mode,
        init_poses=init_poses,
    )
    
    env.start()
    env.join()
