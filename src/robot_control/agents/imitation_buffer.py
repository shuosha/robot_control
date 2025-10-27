from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def _norm_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit length. Accepts shape (..., 4)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n


def _quat_geodesic_angle_batch(q_query: np.ndarray, q_db: np.ndarray) -> np.ndarray:
    """
    Geodesic distance between unit quaternions:
    d(q1,q2) = 2*arccos(|dot(q1,q2)|).
    q_query: (4,)  q_db: (N,4)
    returns: (N,)
    """
    q_query = _norm_quat(q_query)
    q_db = _norm_quat(q_db)
    dots = np.abs(np.sum(q_db * q_query[None, :], axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)

def _distance_pair(self, pos1, quat1, grip1, pos2, quat2, grip2, only_pos: bool) -> float:
    if only_pos:
        return self.w_pos * np.linalg.norm(pos1 - pos2)
    # full metric
    pos_term  = np.linalg.norm(pos1 - pos2)
    quat_term = _quat_geodesic_angle_batch(_norm_quat(quat1), _norm_quat(quat2))
    grip_term = np.abs(float(grip1) - float(grip2))
    return self.w_pos * pos_term + self.w_quat * float(quat_term) + self.w_grip * float(grip_term)

class ImitationBuffer:
    """
    Minimal data buffer for obs->action NN lookup with episodic indexing for chunks.

    Directory layout (example):
    data_root/
      episode_0000/
        robot/
          000000.json
          000001.json
          ...
      episode_0001/
        robot/
          000000.json
          ...

    Each JSON contains (for both 'obs' and 'action'):
      {key}.ee_pos (len 3), {key}.ee_quat (len 4), {key}.gripper_qpos (scalar or len 1)
    """

    def __init__(
        self,
        data_root: str | Path,
        w_pos: float = 1.0,
        w_quat: float = 0.0,
        w_grip: float = 0.0,
        reuse_dist_threshold: Optional[float] = 1e-3,
    ):
        """
        Args
        - data_root: root folder containing episode_* subdirs.
        - w_pos, w_quat, w_grip: weights for the distance metric components.
        - reuse_dist_threshold: if new obs distance to last obs < threshold, reuse last action.
          Set to None to disable reuse heuristic.
        """
        self.root = Path(data_root)
        self.w_pos = float(w_pos)
        self.w_quat = float(w_quat)
        self.w_grip = float(w_grip)
        self.reuse_dist_threshold = reuse_dist_threshold

        # Flat arrays for fast vectorized distance
        self.obs_pos: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.obs_quat: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self.obs_grip: np.ndarray = np.empty((0, 1), dtype=np.float64)

        self.act_pos: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.act_quat: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self.act_grip: np.ndarray = np.empty((0, 1), dtype=np.float64)

        # For chunking
        self.index_episode: List[int] = []
        self.index_t: List[int] = []
        self.episode_offsets: Dict[int, Tuple[int, int]] = {}  # ep_id -> (start_idx, length)

        # Cache for "don't re-query every tiny update"
        self._last_query_obs: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (pos, quat, grip)
        self._last_query_result: Optional[Dict[str, Any]] = None

        self._load_all()

    # ---------------------------- Loading ----------------------------

    def _load_all(self) -> None:
        """Parse all episodes and build flat arrays + episodic index."""
        episodes = sorted(self.root.glob("episode_*"))
        flat_obs_pos, flat_obs_quat, flat_obs_grip = [], [], []
        flat_act_pos, flat_act_quat, flat_act_grip = [], [], []

        cursor = 0
        for ep_id, ep_path in enumerate(episodes):
            robot_dir = ep_path / "robot"
            if not robot_dir.exists():
                continue
            # Sort by integer stem (e.g., "000123" -> 123)
            step_files = sorted(robot_dir.glob("*.json"), key=lambda p: int(p.stem))
            start_idx = cursor
            for t, f in enumerate(step_files):
                try:
                    with open(f, "r") as fh:
                        data = json.load(fh)
                except Exception:
                    continue

                # Extract obs
                try:
                    o_pos = np.asarray(data["obs.ee_pos"], dtype=np.float64).reshape(3)
                    o_quat = np.asarray(data["obs.ee_quat"], dtype=np.float64).reshape(4)
                    o_grip_raw = data["obs.gripper_qpos"]
                    o_grip = np.asarray([o_grip_raw], dtype=np.float64).reshape(1)
                except Exception:
                    # Skip if malformed
                    continue

                # Extract action
                try:
                    a_pos = np.asarray(data["action.ee_pos"], dtype=np.float64).reshape(3)
                    a_quat = np.asarray(data["action.ee_quat"], dtype=np.float64).reshape(4)
                    a_grip_raw = data["action.gripper_qpos"]
                    a_grip = np.asarray([a_grip_raw], dtype=np.float64).reshape(1)
                except Exception:
                    continue

                flat_obs_pos.append(o_pos)
                flat_obs_quat.append(_norm_quat(o_quat))
                flat_obs_grip.append(o_grip)

                flat_act_pos.append(a_pos)
                flat_act_quat.append(_norm_quat(a_quat))
                flat_act_grip.append(a_grip)

                self.index_episode.append(ep_id)
                self.index_t.append(t)
                cursor += 1

            self.episode_offsets[ep_id] = (start_idx, cursor - start_idx)

        if cursor == 0:
            raise RuntimeError(f"No valid steps found under {self.root}")

        # Stack
        self.obs_pos = np.vstack(flat_obs_pos)
        self.obs_quat = np.vstack(flat_obs_quat)
        self.obs_grip = np.vstack(flat_obs_grip)

        self.act_pos = np.vstack(flat_act_pos)
        self.act_quat = np.vstack(flat_act_quat)
        self.act_grip = np.vstack(flat_act_grip)

        self.act_mat = np.hstack([self.act_pos, self.act_quat, self.act_grip])  # (N, 8)

    # ---------------------------- Distance ----------------------------

    def _distance_to_all(self, obs, only_pos: bool = True) -> np.ndarray:
        pos = np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)
        if only_pos:
            return self.w_pos * np.linalg.norm(self.obs_pos - pos[None, :], axis=1)

        quat = _norm_quat(np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4))
        grip = float(np.asarray(obs["gripper_qpos"], dtype=np.float64).reshape(()))
        pos_term  = np.linalg.norm(self.obs_pos - pos[None, :], axis=1)
        quat_term = _quat_geodesic_angle_batch(quat, self.obs_quat)
        grip_term = np.abs(self.obs_grip.squeeze(-1) - grip)
        return self.w_pos * pos_term + self.w_quat * quat_term + self.w_grip * grip_term

    # ---------------------------- Public API ----------------------------

    def query(
        self,
        obs: Dict[str, np.ndarray | float | List[float]],
        top_k: int = 1,
        action_chunk_len: int = 1,
        only_pos: bool = True,
    ) -> Dict[str, Any]:
        """
        Nearest-neighbor query.
        - top_k: return the k best matches (multimodal).
        - action_chunk_len: if >1, return that many consecutive actions starting at the matched time index within the same episode (clipped at episode end).

        Returns a dict:
        {
          "indices": np.ndarray (k,),
          "distances": np.ndarray (k,),
          "actions":  list of np.ndarrays, each shape (chunk_len, 8) in order [pos(3), quat(4), grip(1)]
                      If chunk_len==1 and k==1, returns a single np.ndarray shape (8,).
          "meta":     list of dicts with {"episode_id": int, "t0": int}
        }
        """
        # Reuse heuristic
        if self.reuse_dist_threshold is not None and self._last_query_obs is not None:
            last_pos, last_quat, last_grip = self._last_query_obs
            cur_pos  = np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)
            cur_quat = _norm_quat(np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4))
            cur_grip = float(np.asarray(obs["gripper_qpos"], dtype=np.float64).reshape(()))
            d = self._distance_pair(last_pos, last_quat, last_grip, cur_pos, cur_quat, cur_grip, only_pos)
            if d < self.reuse_dist_threshold and self._last_query_result is not None:
                return self._last_query_result

        dists = self._distance_to_all(obs, only_pos=only_pos)  # (N,)
        k = min(int(top_k), dists.shape[0])
        nn_idx = np.argpartition(dists, kth=k - 1)[:k]
        # Sort the chosen k by distance
        order = np.argsort(dists[nn_idx])
        nn_idx = nn_idx[order]
        nn_d = dists[nn_idx]

        actions_out = []
        metas = []
        for idx in nn_idx:
            ep = self.index_episode[idx]
            t0 = self.index_t[idx]

            base = self.episode_offsets[ep][0] + t0
            max_len = min(action_chunk_len, self.episode_offsets[ep][1] - t0)
            sl = slice(base, base + max_len)
            chunk = self.act_mat[sl]  # (max_len, 8)
            
            actions_out.append(chunk)
            metas.append({"episode_id": ep, "t0": t0})

        result: Dict[str, Any] = {
            "indices": nn_idx,
            "distances": nn_d,
            "actions": actions_out,
            "meta": metas,
        }

        # Update reuse cache
        self._last_query_obs = (
            np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3),
            _norm_quat(np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4)),
            float(np.asarray(obs["gripper_qpos"], dtype=np.float64).reshape(())),
        )
        self._last_query_result = result
        return result
