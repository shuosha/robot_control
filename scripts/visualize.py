import argparse
import sys
from robot_control.visualization.visualize_trajectories import visualize_robot_trajectory_3D, visualize_robot_timeseries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--mode', type=str, default='timeseries', choices=['3d', 'timeseries'])
    parser.add_argument('--eps', type=int, default=0)
    args = parser.parse_args()

    # assert args.name != '', "Please provide a name for the experiment"
    # npz_path = f"logs/teleop/{args.name}_processed/debug/robot_trajectories.npz"
    npz_path = "logs/sim_replay/1027_adm_v0/sim_trajs.npz"
    if args.mode == '3d':
        out_path = visualize_robot_trajectory_3D(npz_path=npz_path, eps_idx=args.eps)
        print(f"3D trajectory plot saved to: {out_path}")
    elif args.mode == 'timeseries':
        out_path, _, _ = visualize_robot_timeseries(npz_path=npz_path, eps_idx=args.eps)
        print(f"Time-series plot saved to: {out_path}")


