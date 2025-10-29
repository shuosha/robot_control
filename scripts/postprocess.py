import argparse
import sys

from robot_control.data_processing.postprocesser import synchronize_timesteps, load_robot_trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('--mode', type=str, default='synchronize', choices=['synchronize', 'load_robot_data'])
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    if args.mode == 'synchronize':
        num_cams = 2
        dir_name = "teleop"
        synchronize_timesteps(name=args.name, num_cams=num_cams, dir_name=dir_name)
    elif args.mode == 'load_robot_data':
        dir_name = "teleop"
        load_robot_trajectories(name=args.name, dir_name=dir_name)
    else:
        raise NotImplementedError(f"Mode {args.mode} not supported yet")