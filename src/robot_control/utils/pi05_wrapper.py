from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

import cv2
import numpy as np

class PI05Wrapper():
    def __init__(self, prompt, cfg_name="pi05_droid", checkpoint_path="gs://openpi-assets/checkpoints/pi05_droid"):
        self.cfg_name = cfg_name
        config = _config.get_config(cfg_name)
        checkpoint_dir = download.maybe_download(checkpoint_path)
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        if self.cfg_name == "pi05_droid":
            self.obs_dict = {
                "observation/exterior_image_1_left": None,      # np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "observation/wrist_image_left": None,           # np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "observation/joint_position": None,             # np.random.rand(7),
                "observation/gripper_position": None,           # np.random.rand(1),
                "prompt": prompt,                                 # "do something",
            }
            self.action_horizon = 15
        else:
            raise NotImplementedError(f"Config {cfg_name} not supported yet")
        
        self.step_idx = 0
        self.action_queue = []

    def transform_obs(self, env_obs: dict) -> dict:
        """
        env_obs: {
            'rgbs': List[np.ndarray],  # list of (H, W, 3) uint8
            'depths': List[np.ndarray],  # list of (H, W) uint16
            'trans_out': np.ndarray,  # (4, 4) float64
            'qpos_out': np.ndarray,  # (7, ) float64
            'gripper_out': np.ndarray,  # (1, ) float64
        }
        """

        front_rgb = cv2.resize(env_obs['rgbs'][0], (224, 224), interpolation=cv2.INTER_LINEAR)
        wrist_rgb = cv2.resize(env_obs['rgbs'][1], (224, 224), interpolation=cv2.INTER_LINEAR)
        qpos = env_obs['qpos_out'].astype(np.float32)
        gripper = ((800.0 - np.clip(env_obs['gripper_out'], 0.0, 800.0)) / 800.0).astype(np.float32)
        gripper_binary = np.array([1.0], dtype=np.float32) if gripper > 0.5 else np.array([0.0], dtype=np.float32)

        if self.cfg_name == "pi05_droid":
            self.obs_dict["observation/exterior_image_1_left"] = front_rgb
            self.obs_dict["observation/wrist_image_left"] = wrist_rgb
            self.obs_dict["observation/joint_position"] = qpos
            self.obs_dict["observation/gripper_position"] = gripper_binary
        else:
            raise NotImplementedError(f"Config {self.cfg_name} not supported yet")
        
        return self.obs_dict

    def untransform_action(self, action: np.ndarray) -> np.ndarray:
        """
        action space depending on config
        return: xarm 8d qpos action
        """
        if self.cfg_name == "pi05_droid":
            # action: (8, ) float32, first 7 are abs qpos, last is gripper open/close
            qpos_cmd = action[:7]
            gripper_cmd = action[7]

            # gripper command
            if gripper_cmd > 0.5:
                gripper_value = 1.0  # close
            else:
                gripper_value = 0.0  # open
            full_action = np.concatenate([qpos_cmd, np.array([gripper_value], dtype=np.float32)], axis=0)
            return full_action
        else:
            raise NotImplementedError(f"Config {self.cfg_name} not supported yet")

    def get_action(self, observation: dict) -> np.ndarray:
        if len(self.action_queue) == 0:
            obs_transformed = self.transform_obs(observation)
            action_chunk = self.policy.infer(obs_transformed)["actions"]
            self.action_queue = [self.untransform_action(action_chunk[i]) for i in range(self.action_horizon)]

        action = self.action_queue.pop(0).astype(np.float32)
        assert len(action) == 8, "xarm7 requires 8d action"

        return action