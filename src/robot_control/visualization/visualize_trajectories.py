from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    eps_idx: int = 0,
    out_path: Optional[Union[str, Path]] = None,
    recenter: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Plot 2D time-series for EEF positions (obs vs action), and (optionally) force xyz if present.

    Expected NPZ keys per episode prefix "episode_xxx":
      - "{ep}/obs.eef_pos"     -> (T, 3)
      - "{ep}/action.eef_pos"  -> (T, 3)
      - Optional "{ep}/obs.force"  -> (T, 6)  [first 3 = force xyz, last 3 = torque rpy]

    Behavior
    --------
    - Produces a single PNG with 3 vertically stacked subplots (X/Y/Z position vs time),
      plotting both obs and action on each axis.
    - If force is present, also produces a second PNG with 3 vertically stacked subplots
      (Fx/Fy/Fz vs time).
    - If `recenter=True`, subtracts the first obs position so time 0 = (0,0,0).
    - Returns (pos_plot_path, force_plot_path_or_None).
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    # discover episodes
    ep_keys = sorted({k.split("/", 1)[0] for k in data.files})
    if not ep_keys:
        raise ValueError(f"No episodes found in {npz_path}")
    if not (0 <= eps_idx < len(ep_keys)):
        raise IndexError(f"eps_idx {eps_idx} out of range (0..{len(ep_keys)-1})")
    ep = ep_keys[eps_idx]

    # required arrays
    try:
        obs_pos = np.asarray(data[f"{ep}/obs.eef_pos"], dtype=np.float32)
        act_pos = np.asarray(data[f"{ep}/action.eef_pos"], dtype=np.float32)
    except KeyError as e:
        raise KeyError(f"Missing required key in NPZ: {e}") from e

    if obs_pos.ndim != 2 or obs_pos.shape[1] != 3:
        raise ValueError(f"{ep}/obs.eef_pos must be (T,3); got {obs_pos.shape}")
    if act_pos.ndim != 2 or act_pos.shape[1] != 3:
        raise ValueError(f"{ep}/action.eef_pos must be (T,3); got {act_pos.shape}")
    if len(obs_pos) != len(act_pos):
        raise ValueError(f"Obs and action lengths differ: {len(obs_pos)} vs {len(act_pos)}")

    # optional force: try "{ep}/obs.force" first, fall back to plain "obs.force" if one-episode NPZ
    force_arr = None
    force_key_candidates = [f"{ep}/obs.force", "obs.force"]
    for k in force_key_candidates:
        if k in data.files:
            arr = np.asarray(data[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 6:
                if len(arr) != len(obs_pos):
                    # We allow force length mismatch but warn by aligning to min length.
                    T = min(len(arr), len(obs_pos))
                    arr = arr[:T]
                    obs_pos = obs_pos[:T]
                    act_pos = act_pos[:T]
                force_arr = arr
                break

    # recenter (positions only)
    if recenter:
        origin = obs_pos[0].copy()
        obs_pos = obs_pos - origin
        act_pos = act_pos - origin

    # time axis (index-based; replace with real timestamps if available)
    T = len(obs_pos)
    t = np.arange(T)

    # output paths
    if out_path is None:
        out_path = npz_path.parent / f"{ep}_traj_timeseries.png"
    out_path = Path(out_path)
    out_path_force = out_path.with_name(out_path.stem + "_force.png")
    out_path_torque = out_path.with_name(out_path.stem + "_torque.png")

    # Use non-interactive backend for headless saving
    matplotlib.use("Agg")

    # -------------------------
    # Positions: 3 stacked axes
    # -------------------------
    fig_pos, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    for i, ax in enumerate(axs):
        ax.plot(t, obs_pos[:, i], label="obs", linewidth=1.8)
        ax.plot(t, act_pos[:, i], label="action", linewidth=1.8, linestyle="--")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"EEF Position Timeseries — {ep}")

    axs[-1].set_xlabel("Time step")
    axs[0].legend(loc="upper right", frameon=True)
    fig_pos.savefig(out_path, dpi=150)
    plt.close(fig_pos)
    print(f"[saved] {out_path}")

    # -------------------------
    # Forces (optional)
    # -------------------------
    force_path = None
    torque_path = None
    if force_arr is not None:
        fig_f, axs_f = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
        flabels = ["Fx (N)", "Fy (N)", "Fz (N)"]
        for i, ax in enumerate(axs_f):
            ax.plot(t, force_arr[:, i], linewidth=1.8)
            ax.set_ylabel(flabels[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"EEF Force Timeseries — {ep} (xyz only)")

        axs_f[-1].set_xlabel("Time step")
        fig_f.savefig(out_path_force, dpi=150)
        plt.close(fig_f)
        print(f"[saved] {out_path_force}")
        force_path = out_path_force

        fig_f, axs_f = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
        flabels = ["Tau_r (N)", "Tau_p (N)", "Tau_y (N)"]
        for i, ax in enumerate(axs_f):
            ax.plot(t, force_arr[:, -i], linewidth=1.8)
            ax.set_ylabel(flabels[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"EEF Torque Timeseries — {ep} (rpy only)")

        axs_f[-1].set_xlabel("Time step")
        fig_f.savefig(out_path_torque, dpi=150)
        plt.close(fig_f)
        print(f"[saved] {out_path_torque}")
        torque_path = out_path_torque

    return out_path, force_path, torque_path
