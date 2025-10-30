import argparse
import sys
from robot_control.visualization.visualize_trajectories import visualize_robot_trajectory_3D, visualize_robot_timeseries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='ts', choices=['3d', 'ts'])
    parser.add_argument('--eps', type=int, default=0)
    args = parser.parse_args()

    assert args.npz_path != '', "Please provide a path to the NPZ file"

    if args.mode == '3d':
        out_path = visualize_robot_trajectory_3D(npz_path=args.npz_path, eps_idx=args.eps)
        print(f"3D trajectory plot saved to: {out_path}")
    elif args.mode == 'ts':
        visualize_robot_timeseries(npz_path=args.npz_path)


