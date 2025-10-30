from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Union, Tuple


def visualize_robot_trajectory_3D(
    npz_path: str | Path,
    eps_idx: int = 0,
    out_path: Optional[str | Path] = None,
    interactive: bool = False,
) -> Path:
    """
    Plot 3D EEF trajectories (obs/action) from a single NPZ that stores:
      "episode_xxx/obs.eef_pos"     → (T,3)
      "episode_xxx/action.eef_pos"  → (T,3)
    - Recenter so the first obs position is at (0,0,0)
    - Blue gradient for obs, red gradient for action
    - 's' to save, 'q' to save & quit
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    # episodes are prefixes before the first '/'
    ep_keys = sorted({k.split("/", 1)[0] for k in data.files})
    if not ep_keys:
        raise ValueError(f"No episodes found in {npz_path}")
    if not (0 <= eps_idx < len(ep_keys)):
        raise IndexError(f"eps_idx {eps_idx} out of range (0..{len(ep_keys)-1})")

    ep = ep_keys[eps_idx]
    # --- read positions from keys ---
    try:
        obs_pos = np.asarray(data[f"{ep}/obs.eef_pos"], dtype=np.float32)
        act_pos = np.asarray(data[f"{ep}/action.eef_pos"], dtype=np.float32)
    except KeyError as e:
        raise KeyError(f"Missing required key in NPZ: {e}") from e
    if obs_pos.ndim != 2 or obs_pos.shape[1] != 3 or act_pos.ndim != 2 or act_pos.shape[1] != 3:
        raise ValueError("Expected obs.eef_pos and action.eef_pos to be shaped (T,3)")

    # Recenter to first obs point
    origin = obs_pos[0].copy()
    obs_pos = obs_pos - origin
    act_pos = act_pos - origin

    # Default output path
    if out_path is None:
        out_path = npz_path.parent / f"{ep}_traj3d.png"
    out_path = Path(out_path)

    if not interactive:
        matplotlib.use("Agg")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Temporal gradients
    t_obs = np.linspace(0, 1, len(obs_pos))
    t_act = np.linspace(0, 1, len(act_pos))
    cmap_obs, cmap_act = plt.cm.Blues, plt.cm.Reds

    ax.scatter(*obs_pos.T, c=cmap_obs(t_obs), s=6, alpha=0.95, depthshade=False)
    ax.scatter(*act_pos.T, c=cmap_act(t_act), s=6, alpha=0.95, depthshade=False)

    # Mark global origin (after recentering it's at 0,0,0)
    ax.scatter([0], [0], [0], s=60, marker="x", c="k")

    # Legend
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor=plt.cm.Blues(0.85), markeredgecolor='none', label='obs pos'),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor=plt.cm.Reds(0.85), markeredgecolor='none', label='action pos'),
        Line2D([0], [0], marker='x', linestyle='None', markersize=8, color='k', label='origin'),
    ]

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"EEF 3D Trajectory — {ep}")
    ax.legend(handles=handles, loc='best')

    # Equal scaling
    all_pts = np.vstack([obs_pos, act_pos])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    centers, span = (mins + maxs) / 2, float((maxs - mins).max() or 1e-6)
    ax.set_xlim(centers[0] - span/2, centers[0] + span/2)
    ax.set_ylim(centers[1] - span/2, centers[1] + span/2)
    ax.set_zlim(centers[2] - span/2, centers[2] + span/2)
    fig.tight_layout()

    def _save(path: Path):
        fig.savefig(path, dpi=150)
        print(f"[saved] {path}")

    if not interactive:
        _save(out_path); plt.close(fig); return out_path

    saved = {"v": False}
    def on_key(e):
        if e.key == "s":
            _save(out_path); saved["v"] = True
        elif e.key == "q":
            _save(out_path); saved["v"] = True; plt.close(fig)

    def on_close(_):
        if not saved["v"]:
            _save(out_path)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    plt.show(); plt.close(fig)
    return out_path

def visualize_robot_timeseries(
    npz_path: Union[str, Path],
    recenter: bool = False,
):
    """
    Plot time-series for all episodes in NPZ:
      - Positions (obs vs action)
      - Forces (Fx,Fy,Fz)
      - Torques (Tx,Ty,Tz)
    Saves plots into three folders: position_ts, force_ts, torque_ts.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)
    ep_keys = sorted({k.split("/", 1)[0] for k in data.files})
    if not ep_keys:
        raise ValueError(f"No episodes found in {npz_path}")

    # Create output folders
    parent = npz_path.parent
    out_dirs = {
        "pos": parent / "position_ts",
        "force": parent / "force_ts",
        "torque": parent / "torque_ts",
    }
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)

    def plot_xyz(arr, labels, title, save_path):
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
        for i, ax in enumerate(axs):
            ax.plot(np.arange(len(arr)), arr[:, i], linewidth=1.8)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
        axs[0].set_title(title)
        axs[-1].set_xlabel("Timestep")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {save_path}")

    for ep in ep_keys:
        try:
            obs = np.asarray(data[f"{ep}/obs.eef_pos"], dtype=np.float32)
            act = np.asarray(data[f"{ep}/action.eef_pos"], dtype=np.float32)
        except KeyError:
            print(f"[warn] Missing obs/action for {ep}, skipping.")
            continue

        if recenter:
            origin = obs[0].copy()
            obs -= origin
            act -= origin

        # Position plots
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
        for i, lbl in enumerate(["X (m)", "Y (m)", "Z (m)"]):
            axs[i].plot(obs[:, i], label="obs", linewidth=1.8)
            axs[i].plot(act[:, i], label="action", linestyle="--", linewidth=1.8)
            axs[i].set_ylabel(lbl)
            axs[i].grid(True, alpha=0.3)
        axs[0].set_title(f"EEF Position Timeseries — {ep}")
        axs[-1].set_xlabel("Timestep")
        axs[0].legend()
        fig.savefig(out_dirs["pos"] / f"{ep}_position.png", dpi=150)
        plt.close(fig)

        # Force/torque if available
        fkey = f"{ep}/obs.force"
        if fkey not in data:
            continue

        force_arr = np.asarray(data[fkey], dtype=np.float32)
        if force_arr.ndim != 2 or force_arr.shape[1] != 6:
            print(f"[warn] Bad shape {force_arr.shape} for {fkey}, skipping.")
            continue

        fx = force_arr[:, :3]
        tx = force_arr[:, 3:]
        plot_xyz(fx, ["Fx (N)", "Fy (N)", "Fz (N)"], f"EEF Force — {ep}", out_dirs["force"] / f"{ep}_force.png")
        plot_xyz(tx, ["Tx (Nm)", "Ty (Nm)", "Tz (Nm)"], f"EEF Torque — {ep}", out_dirs["torque"] / f"{ep}_torque.png")

    print(f"All plots saved under:\n  {out_dirs['pos']}\n  {out_dirs['force']}\n  {out_dirs['torque']}")