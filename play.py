import os
from pathlib import Path
import argparse
import multiprocess as mp
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from robot_control.utils.utils import get_root
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))

import torch
import json
import numpy as np
from experiments.real_world.modules_teleop.robot_teleop_env import RobotTeleopEnv

def get_last_timestep(dones_row: torch.Tensor) -> range:
    """Timesteps up to and including the terminal step."""
    idxs = torch.nonzero(dones_row, as_tuple=False)
    end = (idxs[0].item() + 1) if idxs.numel() else dones_row.shape[0]
    return end

def collect_robot_data(robot_dir: str, key: str = "action", pusht=False):
    robot_path = Path(robot_dir)
    json_files = sorted(robot_path.glob("*.json"))  # 000000.json, 000001.json, ...
    
    traj = []
    for f in json_files:
        with open(f, "r") as j:
            data = json.load(j)
        if pusht:
            if key == "action":
                vec = data["action.xy"]
            elif key == "obs":
                vec = (data[f"{key}.ee_pos"]
                + data[f"{key}.ee_quat"]
                + [0.0])
        else:
            # concatenate into a single vector
            vec = (
                data[f"{key}.ee_pos"]
                + data[f"{key}.ee_quat"]
                + [data[f"{key}.gripper_qpos"]]
            )
        traj.append(vec)

    return np.array(traj, dtype=np.float32)  # shape (T, 8)

if __name__ == '__main__':
    # cv2 encounter error when using multi-threading, use tk instead
    # cv2.setNumThreads(cv2.getNumberOfCPUs())
    # cv2.namedWindow("real env monitor", cv2.WINDOW_NORMAL)

    mp.set_start_method('spawn')  # type: ignore
    # torch.multiprocess.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--save_json', action='store_true', default=True)
    parser.add_argument('--task_name', type=str, required=True, choices=['insert_rope', 'pack_sloth', 'pusht'])
    parser.add_argument('--robot', type=str, default='xarm', choices=['xarm', 'aloha'])
    parser.add_argument('--input_mode', type=str, default='teleop', choices=['teleop', 'replay', 'policy'])
    parser.add_argument('--checkpoint_path', type=str, default='', required=False)
    parser.add_argument('--episode_dir', type=str, default=os.path.join(root, "data", "replay_cache"), required=False)
    parser.add_argument('--use_Pi0', action='store_true', default=False)
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    if args.input_mode == "replay":
        assert args.episode_dir != '', "Please provide an eps directory for the replay data"

        action_traj = collect_robot_data(os.path.join(args.episode_dir, "robot"), key="action", pusht=args.pusht)  # shape: (T, 8)
        obs_traj = collect_robot_data(os.path.join(args.episode_dir, "robot"), key="obs", pusht=args.pusht)  # shape: (T, 8)

        init_pose = obs_traj[0]  # initial observation as starting pose

        env = RobotTeleopEnv(
            mode='3D',
            exp_name=args.name,
            resolution=(848, 480),
            capture_fps=30,
            record_fps=30,
            perception_process_func=None,
            use_xarm=True if args.robot == 'xarm' else False,
            use_aloha=True if args.robot == 'aloha' else False,
            use_gello=True, 
            bimanual=False if args.robot == 'xarm' else True, # NOTE: hardcoded for xarm and aloha
            gripper_enable=True if not args.pusht else False,
            pusht_teleop=args.pusht,
            data_dir="data",
            debug=True,
            save_robot_json=args.save_json,
            action_traj=action_traj,
            input_mode=args.input_mode,
            init_pose=init_pose,
        )

        # print("Initial pose:", init_pose)
    elif args.input_mode == "teleop":
        env = RobotTeleopEnv(
            mode='3D',
            exp_name=args.name,
            resolution=(848, 480),
            capture_fps=30,
            record_fps=30,
            perception_process_func=None,
            use_xarm=True if args.robot == 'xarm' else False,
            use_aloha=True if args.robot == 'aloha' else False,
            use_gello=True, 
            bimanual=False if args.robot == 'xarm' else True, # NOTE: hardcoded for xarm and aloha
            gripper_enable=True if args.task_name != "pusht" else False,
            pusht_teleop=args.task_name == "pusht",
            data_dir="data",
            debug=True,
            save_robot_json=args.save_json,
            input_mode=args.input_mode,
        )
    elif args.input_mode == "policy":
        assert args.checkpoint_path != '', "Please provide a checkpoint path for the policy"
        env = RobotTeleopEnv(
            mode='3D',
            exp_name=args.name,
            resolution=(848, 480),
            capture_fps=30,
            record_fps=30,
            perception_process_func=None,
            use_xarm=True,
            use_gello=True,
            bimanual=args.bimanual,
            gripper_enable=True if args.task_name != "pusht" else False,
            pusht_teleop=args.task_name == "pusht",
            task_name=args.task_name,
            data_dir="policy_rollouts",
            debug=True,
            save_robot_json=args.save_json,
            input_mode=args.input_mode,
            checkpoint_path=args.checkpoint_path,
            use_Pi0=args.use_Pi0,
        )
    else:
        raise ValueError(f"Unknown input mode: {args.input_mode}. Choose from ['teleop', 'replay', 'policy'].")

    env.start()
    env.join()
