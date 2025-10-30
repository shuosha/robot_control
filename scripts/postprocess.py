import argparse
import sys

from robot_control.data_processing.postprocesser import synchronize_timesteps, load_robot_trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='synchronize', choices=['synchronize'])
    args = parser.parse_args()

    assert args.data_path != '', "Please provide a name for the experiment"

    if args.mode == 'synchronize':
        synchronize_timesteps(data_path=args.data_path)
        load_robot_trajectories(data_path=args.data_path)
    else:
        raise NotImplementedError(f"Mode {args.mode} not supported yet")