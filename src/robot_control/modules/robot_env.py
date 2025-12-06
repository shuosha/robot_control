from multiprocessing import shared_memory
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os, sys, contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

from typing import Callable, Sequence, List, Literal, Optional
from enum import Enum
import numpy as np
import multiprocess as mp
import time
import threading
import cv2
with suppress_stdout():
    import pygame
    pygame.init()

import os
import pickle
import transforms3d
import subprocess
from multiprocessing.managers import SharedMemoryManager
from pynput import keyboard
from pathlib import Path
from copy import deepcopy
import json, glob
import torch
from scipy.spatial.transform import Rotation as R
from pprint import pprint
from typing import Union

from robot_control.utils.utils import get_root, mkdir
root: Path = get_root(__file__)

from robot_control.modules.perception import Perception
from robot_control.modules.xarm_controller import XarmController
from robot_control.modules.state_estimator import StateEstimator
from robot_control.utils.udp_util import udpReceiver, udpSender

from robot_control.agents.teleop_keyboard import KeyboardTeleop
from robot_control.agents.action_commander import ActionAgent
from robot_control.agents.nn_buffer import NearestNeighborBuffer

from robot_control.utils.kinematics_utils import KinHelper, trans_mat_to_pos_quat, gripper_raw_to_qpos, _quat_apply, _q_mul, _q_normalize
from robot_control.utils.math import axis_angle_from_quat, quat_mul, quat_conjugate, quat_from_angle_axis, euler_xyz_from_quat, quat_from_euler_xyz, combine_frame_transforms

from robot_control.camera.multi_realsense import MultiRealsense
from robot_control.camera.single_realsense import SingleRealsense
from robot_control.modules.common.communication import RGB_PORT, DEPTH_PORT

import torch
import sys
import yaml
import numpy as np
from rl_games.algos_torch.players import PpoPlayerContinuous
# try:
from gymnasium.spaces import Box
# except ImportError:
#     from gym.spaces import Box  # fallback if gymnasium is not installed

# from robot_control.utils.pi05_wrapper import PI05Wrapper

# TODO: import policy inference wrapper

XARM7_LEFT_IP = "192.168.1.196"
XARM7_RIGHT_IP = "192.168.1.224"
UF850_IP = "192.168.1.220"

class EnvEnum(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3

class RobotEnv(mp.Process):
    def __init__(
        self,
        # --------------------- Logging ---------------------
        debug: int = 0,  # 0: silent, 1: info, 2: debug, 3: verbose
        exp_name: str = "recording",
        data_dir: Path = Path("data"),

        # --------------------- Cameras ---------------------
        realsense: Union[MultiRealsense, SingleRealsense, None] = None,
        shm_manager: Union[SharedMemoryManager, None] = None,
        serial_numbers: Union[Sequence[str], None] = None,
        resolution: tuple[int, int] = (848, 480),
        capture_fps: int = 30,
        record_fps: Union[int, None] = 0,
        record_time: Union[float, None] = 60 * 10,  # seconds
        enable_depth: bool = True,
        enable_color: bool = True,

        # --------------------- Perception ---------------------
        perception: Union[Perception, None] = None,
        perception_process_func: Union[Callable, None] = None,  # identity if None
        foundation_pose_dir: Union[Path, None] = None,

        # --------------------- Robot ---------------------
        robot_name: Literal["xarm7", "aloha", "uf850"] = "xarm7",
        control_mode: Literal["position_control", "velocity_control"] = "position_control",
        admittance_control: bool = False,
        ema_factor: float = 0.7,
        bimanual: bool = False,
        robot_ip: List[str] = [XARM7_LEFT_IP],
        gripper_enable: bool = False,
        calibrate_result_dir: Path = Path("log/latest_calibration"),

        # --------------------- Control ---------------------
        action_receiver: Literal["gello", "keyboard", "policy", "replay", "residual", "residual_offline"] = "gello",
        action_agent_fps: float = 10.0,
        pusht_mode: bool = False,
        init_poses: Union[List[np.ndarray], None] = [],
        action_trajs: Union[np.ndarray, None] = None,
        checkpoint_path: Union[str, None] = None,
    ) -> None:
        """
        Multiprocessing-based robot environment.

        Args:
            --------------------- Logging ---------------------
            exp_name: Name for the experiment or recording session.
            data_dir: Directory where logs and recordings are saved.

            --------------------- Cameras ---------------------
            realsense: Pre-initialized MultiRealsense or SingleRealsense object.
            shm_manager: SharedMemoryManager for camera streams.
            serial_numbers: List of RealSense serial numbers.
            resolution: Camera resolution (width, height).
            capture_fps: Camera capture FPS.
            record_fps: Target recording FPS (0 or None disables).
            record_time: Maximum record time in seconds.
            enable_depth: Enable depth stream.
            enable_color: Enable RGB stream.

            --------------------- Perception ---------------------
            perception: Optional perception module.
            perception_process_func: Function to process perception output (defaults to identity).

            --------------------- Robot ---------------------
            robot_name: Robot type ("xarm7", "aloha", or "uf850").
            bimanual: Whether to use a dual-arm setup.
            robot_ip: List of robot IP addresses (required for all modes).
            gripper_enable: Enable gripper control.
            calibrate_result_dir: Directory containing calibration results.
            debug: Debug level (0 = silent, 1 = debug, 2 = verbose).

            --------------------- Control ---------------------
            action_receiver: Control input mode ("gello", "keyboard", "policy", or "replay").
            init_poses: List of initial robot poses.
            action_trajs: Action trajectories for replay mode (T, action_dim).
            checkpoint_path: Path to policy checkpoint for policy mode.
        """
        super().__init__()

        # ------------ debug level ------------

        # ------------ logging --------------
        self.debug = 0 if debug is None else (2 if debug is True else debug)
        self.exp_name = exp_name
        self.data_dir = Path(data_dir)

        # ------------ cameras (always required for real env) ---------------
        if realsense is not None:
            assert isinstance(realsense, MultiRealsense) or isinstance(realsense, SingleRealsense)
            self.realsense = realsense
            self.serial_numbers = list(self.realsense.cameras.keys())
        else:
            self.realsense = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=serial_numbers,
                resolution=resolution,
                capture_fps=capture_fps,
                enable_depth=enable_depth,
                enable_color=enable_color,
                process_depth=False,
                verbose=self.debug >= EnvEnum.VERBOSE.value
            )
            self.serial_numbers = list(self.realsense.cameras.keys())
        
        # NOTE: hardcoded exposure and white balance for consistency
        # self.realsense.set_exposure(exposure=200, gain=60, depth_exposure=10000, depth_gain=60)  # 100: bright, 60: dark
        # self.realsense.set_white_balance(3800)

        # -- optional -- automatic exposure and white balance
        self.realsense.set_exposure(exposure=None)
        self.realsense.set_white_balance(white_balance=None)

        self.capture_fps = capture_fps
        self.record_fps = record_fps

        # ------------ perception ---------------
        if perception is not None:
            assert isinstance(perception, Perception)
            self.perception = perception
        else:
            self.perception = Perception(
                realsense=self.realsense,
                capture_fps=self.realsense.capture_fps,  # must be the same as realsense capture fps
                record_fps=record_fps,
                record_time=record_time,
                process_func=perception_process_func,
                exp_name=exp_name,
                data_dir=data_dir,
                verbose=self.debug >= EnvEnum.VERBOSE.value)

        # ----------- robot ---------------
        self.bimanual = bimanual
        self.robot_name = robot_name
        self.gripper_enable = gripper_enable

        # initialize controller(s)
        if self.robot_name == "xarm7":
            if self.bimanual:
                assert len(robot_ip) == 2, "Bimanual xArm7 requires two robot IPs"
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip[0],
                    gripper_enable=gripper_enable,
                    control_mode=control_mode,
                    admittance_control=admittance_control,
                    ema_factor=ema_factor,
                    comm_update_fps=action_agent_fps,
                    robot_id=0,
                    verbose=self.debug >= EnvEnum.VERBOSE.value,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip[1],
                    gripper_enable=gripper_enable,
                    control_mode=control_mode,
                    admittance_control=admittance_control,
                    ema_factor=ema_factor,
                    comm_update_fps=action_agent_fps,
                    robot_id=1,
                    verbose=self.debug >= EnvEnum.VERBOSE.value,
                )
                self.xarm_controller = None
            else:                    
                assert len(robot_ip) == 1, "Single xArm7 requires one robot IP"
                self.xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip[0],
                    gripper_enable=gripper_enable,
                    control_mode=control_mode,
                    admittance_control=admittance_control,
                    ema_factor=ema_factor,
                    comm_update_fps=action_agent_fps,
                    robot_id=-1,
                    verbose=self.debug >= EnvEnum.VERBOSE.value,
                )
                self.left_xarm_controller = None
                self.right_xarm_controller = None
        else:
            raise NotImplementedError(f"Robot {self.robot_name} is not implemented yet")

        # ----------- input mode --------------
        self.action_receiver = action_receiver
        self.init_poses = init_poses

        # action agent := module that communicates with the robot
        if self.robot_name == "xarm7":
            # assign variables based on action receiver
            if action_receiver == "replay":
                assert action_trajs is not None, "Action trajectory must be provided for replay mode"
                self.action_trajs = action_trajs
                self.total_trajs = len(self.action_trajs)
                print(f"Replaying {self.total_trajs} trajectories")
            elif action_receiver == "policy":
                # pass
                # assert checkpoint_path is not None, "Checkpoint path must be provided for policy mode"
                # self.policy = PolicyInferenceWrapper(checkpoint_path=checkpoint_path)

                self.policy = PI05Wrapper(prompt="pick up green cube", cfg_name="pi05_droid", checkpoint_path="gs://openpi-assets/checkpoints/pi05_droid")

            # create action agent
            if pusht_mode:
                self.action_agent = ActionAgent(bimanual=self.bimanual, joint_space_dim=2, action_receiver=action_receiver)
            else:
                self.action_agent = ActionAgent(bimanual=self.bimanual, joint_space_dim=8, action_receiver=action_receiver)
        else:
            raise NotImplementedError(f"Teleop for robot {self.robot_name} is not implemented yet")
        
        if foundation_pose_dir is not None:
            self.state_estimator = StateEstimator(
                foundation_pose_dir=foundation_pose_dir,
            )
            self.policy = self.load_actor_mlp(
                agent_yaml="logs/policy/v3_gearmesh_3cm_dmr_relaxed_rew/params/agent.yaml",
                checkpoint_path="logs/policy/v3_gearmesh_3cm_dmr_relaxed_rew/nn/FactoryXarm.pth"
            )

            self.base_actions_agent = NearestNeighborBuffer(
                "logs/teleop/20251119_gearmesh_20_processed/debug/robot_trajectories.npz",
                1,
                min_horizon=300,
                max_horizon=300,
                device="cuda",
                pad=True,
            )
        else:
            self.state_estimator = None        

        # other parameters
        self.state = mp.Manager().dict()  # should be main explict exposed variable to the child class / process
        self._real_alive = mp.Value('b', False)
        self.start_time = 0
        mp.Process.__init__(self)
        self._alive = mp.Value('b', False)

        # image visualization
        img_w, img_h = resolution
        views_per_cam = 2
        num_cams = max(len(self.realsense.serial_numbers),1)

        # Image grid: stacked vertically
        unscaled_width  = 2 * img_w * views_per_cam  # 2 * 848 = 1696
        unscaled_height = 2 * img_h * num_cams       # e.g., 480 * 4 = 1920

        # Get max screen size
        screen_info = pygame.display.Info()
        max_screen_w, max_screen_h = screen_info.current_w, screen_info.current_h

        # Compute scaling factor to fit image into screen
        scale = min(1.0, max_screen_w / unscaled_width, max_screen_h / unscaled_height)

        # Final screen size (fixed for both threads)
        self.screen_width  = int(unscaled_width * scale)
        self.screen_height = int(unscaled_height * scale)

        # Allocate shared memory buffer at scaled size
        self.image_data = mp.Array(
            'B',
            np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8).flatten()
        )

        # TODO add calibration info back in 

    def real_start(self, start_time) -> None:
        self._real_alive.value = True
        print("starting real env")
        
        # Realsense camera setup
        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        time.sleep(2)

        # Perception setup
        if self.perception is not None:
            self.perception.start()
    
        # Robot setup
        if self.robot_name == "xarm7":
            if self.bimanual:
                self.left_xarm_controller.start()
                self.right_xarm_controller.start()
            else:
                self.xarm_controller.start()

        self.action_agent.start()
        
        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)
        
        # get intrinsics
        # intrs = self.get_intrinsics()
        # intrs = np.array(intrs)
        # np.save(root / "logs" / self.data_dir / self.exp_name / "calibration" / "intrinsics.npy", intrs)
        
        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        self._real_alive.value = False
        if self.robot_name == "xarm7":
            if self.bimanual and self.left_xarm_controller.is_controller_alive:
                self.left_xarm_controller.stop()
            if self.bimanual and self.right_xarm_controller.is_controller_alive:
                self.right_xarm_controller.stop()
            if not self.bimanual and self.xarm_controller.is_controller_alive:
                self.xarm_controller.stop()
        if self.perception is not None and self.perception.alive.value:
            self.perception.stop()
        self.realsense.stop(wait=False)

        self.image_display_thread.join()
        self.update_real_state_t.join()

        print("======= Real Env Stopped =======")

    @property
    def real_alive(self) -> bool:
        alive = self._real_alive.value
        if self.perception is not None:
            alive = alive and self.perception.alive.value
        if self.robot_name == "xarm7":
            controller_alive = \
                (self.bimanual and self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive) \
                or (not self.bimanual and self.xarm_controller.is_controller_alive)
            alive = alive and controller_alive
        self._real_alive.value = alive
        return self._real_alive.value

    def _update_perception(self) -> None:
        if self.perception.alive.value:
            if not self.perception.perception_q.empty():
                self.state["perception_out"] = {
                    "value": self.perception.perception_q.get()
                }
        return

    def _update_robot(self) -> None:
        if self.bimanual and self.robot_name == "xarm7":
            if self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive:
                self.state["trans_out"] = {
                    "capture_time": self.left_arm_controller.cur_time_q.value,
                    "left_value": np.array(self.left_xarm_controller.cur_trans_q[:]).reshape(4, 4),
                    "right_value": np.array(self.right_xarm_controller.cur_trans_q[:]).reshape(4, 4),
                }
                self.state["qpos_out"] = {
                    "left_value": np.array(self.left_xarm_controller.cur_qpos_q[:]),
                    "right_value": np.array(self.right_xarm_controller.cur_qpos_q[:])
                }
                self.state["gripper_out"] = {
                    "left_value": np.array(self.left_xarm_controller.cur_gripper_q[:]),
                    "right_value": np.array(self.right_xarm_controller.cur_gripper_q[:])
                }
        elif self.robot_name == "xarm7":
            if self.xarm_controller.is_controller_alive:
                self.state["trans_out"] = {
                    "capture_time": self.xarm_controller.cur_time_q.value,
                    "value": np.array(self.xarm_controller.cur_trans_q[:]).reshape(4, 4)
                }
                self.state["qpos_out"] = {
                    "value": np.array(self.xarm_controller.cur_qpos_q[:])
                }
                self.state["gripper_out"] = {
                    "value": np.array(self.xarm_controller.cur_gripper_q[:])
                }
                self.state["force_out"] = {
                    "value": np.array(self.xarm_controller.cur_force_q[:])
                }
        elif self.robot_name == "aloha": # TODO: add aloha support
            if self.teleop.is_alive():
                self.state["trans_out"] = {
                    "capture_time": self.teleop.cur_time_q.value,
                    "right_value": np.array(self.teleop.cur_trans_R[:]).reshape(4, 4),
                    "left_value": np.array(self.teleop.cur_trans_L[:]).reshape(4, 4)
                }
                self.state["qpos_out"] = {
                    "right_value": np.array(self.teleop.cur_qpos_R[:]),
                    "left_value": np.array(self.teleop.cur_qpos_L[:])
                }
                self.state["gripper_out"] = {
                    "right_value": np.array(self.teleop.cur_gripper_R[:]),
                    "left_value": np.array(self.teleop.cur_gripper_L[:])
                }
        else:
            print("No robot controller is used, skipping robot update")
        return
    
    def _update_command(self) -> None:
        if self.bimanual and self.robot_name == "xarm7":
            raise NotImplementedError("Bimanual command update is not implemented yet")
        elif self.robot_name == "xarm7":
            if self.action_agent.is_alive():
                self.state["action_qpos_out"] = {
                    "capture_time": self.action_agent.cur_time_q.value,
                    "value": np.array(self.action_agent.cur_qpos_comm[:7])
                }
                self.state["action_trans_out"] = {
                    "value": np.array(self.action_agent.cur_eef_trans[:]).reshape(4, 4)
                }
                self.state["action_gripper_out"] = {
                    "value": np.array(self.action_agent.cur_qpos_comm[7:])
                }
        elif self.robot_name == "aloha": # TODO: add aloha support
            if self.teleop.is_alive():
                self.state["action_qpos_out"] = {
                    "capture_time": self.teleop.cur_time_q.value,
                    "right_value": np.array(self.teleop.comm_qpos_R[:]),
                    "left_value": np.array(self.teleop.comm_qpos_L[:])
                }
                self.state["action_trans_out"] = {
                    "right_value": np.array(self.teleop.comm_trans_R[:]).reshape(4, 4),
                    "left_value": np.array(self.teleop.comm_trans_L[:]).reshape(4, 4)
                }
                self.state["action_gripper_out"] = {
                    "right_value": np.array(self.teleop.comm_gripper_R[:]),
                    "left_value": np.array(self.teleop.comm_gripper_L[:])
                }
        else:
            print("No robot command is used, skipping command update")
        return

    def update_real_state(self) -> None:
        while self.real_alive:
            try:
                if self.robot_name == "xarm7" or self.robot_name == "aloha":
                    self._update_robot()
                if self.perception is not None:
                    self._update_perception()
                if self.action_agent is not None:
                    self._update_command()
            except BaseException as e:
                print(f"Error in update_real_state: {e.with_traceback()}")
                break
        print("update_real_state stopped")

    def display_image(self):
        self.image_window = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Image Display Window')
        while self._alive.value:
            # Extract image data from the shared array
            image = np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3))
            pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            # Blit the image to the window
            self.image_window.blit(pygame_image, (0, 0))
            pygame.display.update()

            # Handle events (e.g., close window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Image display window stopped")
                    self.stop()
                    pygame.quit()
                    return

            time.sleep(1 / self.realsense.capture_fps)  # 30 FPS
        print("Image display stopped")

    def start_image_display(self):
        # Start a thread for the image display loop
        self.image_display_thread = threading.Thread(name="display_image", target=self.display_image)
        self.image_display_thread.start()

    def get_qpos_from_action_8d(self, action, curr_qpos) -> np.ndarray:
        '''
        action shape: (8,)
        action[0:3]: x, y, z position of the end effector in the base frame
        action[3:7]: quaternion (w, x, y, z) of the end effector orientation in the base frame
        action[7]: gripper qpos (0 for open, 0.8 for close)
        '''
        assert action.shape[0] == 8, "Action shape must be (8,) for robot control"
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

        pos = action_np[0:3]
        quat_wxyz = action_np[3:7] # quaternion (w, x, y, z)
        quat_xyzw = np.roll(quat_wxyz, -1)  # convert to (x, y, z, w) format
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)  # euler angles (radians)
        cartesian_goal = np.concatenate([pos, euler], axis=0)  # (6,)
        gripper_qpos = action_np[7:] # (1,)
        goal_qpos = self.kin_helper.compute_ik_sapien(initial_qpos=curr_qpos, cartesian=cartesian_goal)

        return np.concatenate([goal_qpos, gripper_qpos], axis=0)  # (8,)

    def set_robot_initial_pose(self, init_pose: Sequence[float], fps=30.0) -> None:
        if self.bimanual:
            raise NotImplementedError("Bimanual replay is not implemented yet")
        else:
            print("Resetting robot to initial pose (deg):", init_pose)
            init_pose *= np.pi/180
            print("Initial pose (radians):", init_pose)
            init_pose = init_pose.tolist()

            assert self.alive, "Environment must be running to set initial pose"
            self.xarm_controller.teleop_activated.value = True
            for _ in range(200):
                tic = time.time()
                self.action_agent.command[:] = init_pose
                time.sleep(max(0, 1 / fps - (time.time() - tic)))
            time.sleep(1)
            print("Initial pose set")

            if self.action_receiver == "replay" or self.action_receiver == "policy" or self.action_receiver == "residual_offline":
                self.action_agent.record_start.value = True
            elif self.action_receiver == "keyboard" or self.action_receiver == "gello" or self.action_receiver == "residual":
                self.xarm_controller.teleop_activated.value = False

    def store_robot_data(self, trans_out, qpos_out, gripper_out, action_qpos_out, action_trans_out, action_gripper_out, robot_obs_record_dir, robot_action_record_dir, force_out=None) -> None:
        res_obs = {}
        if self.bimanual:
            # store eef poses
            qpos_L = qpos_out["left_value"]
            trans_L = trans_out["left_value"]

            qpos_R = qpos_out["right_value"]
            trans_R = trans_out["right_value"]

            pos_l, rot_l = trans_mat_to_pos_quat(trans_L)
            res_obs['obs.qpos.left'] = qpos_L.tolist()
            res_obs['obs.ee_pos.left'], res_obs['obs.ee_quat.left'] = pos_l.tolist(), rot_l.tolist()

            pos_r, rot_r = trans_mat_to_pos_quat(trans_R)
            res_obs['obs.qpos.right'] = qpos_R.tolist()
            res_obs['obs.ee_pos.right'], res_obs['obs.ee_quat.right'] = pos_r.tolist(), rot_r.tolist()

            if self.gripper_enable:
                left_gripper = gripper_out["left_value"]
                right_gripper = gripper_out["right_value"]
                res_obs['obs.gripper_qpos.left'] = left_gripper.tolist() # edited to _qpos
                res_obs['obs.gripper_qpos.right'] = right_gripper.tolist()
        else:
            ee2base = trans_out["value"]  # (4, 4)
            pos, quat = trans_mat_to_pos_quat(ee2base)
            res_obs["obs.ee_pos"], res_obs["obs.ee_quat"] = pos.tolist(), quat.tolist()

            qpos = qpos_out["value"]
            res_obs['obs.qpos'] = qpos.tolist()

            if self.gripper_enable:
                gripper = gripper_out["value"][0]
                res_obs['obs.gripper_qpos'] = gripper_raw_to_qpos(gripper) # NOTE: not np array for xarm

            if force_out is not None:
                force = force_out["value"]
                res_obs['obs.force'] = force.tolist()

        with open(robot_obs_record_dir / f"{trans_out['capture_time']:.3f}.json", 'w') as f:
            json.dump(res_obs, f, indent=4)
        
        # save action in a different file
        res_action = {}

        # add action
        if self.bimanual:
            action_qpos_L = action_qpos_out["left_value"]
            action_trans_L = action_trans_out["left_value"]

            action_qpos_R = action_qpos_out["right_value"]
            action_trans_R = action_trans_out["right_value"]

            pos_l, rot_l = trans_mat_to_pos_quat(action_trans_L)
            res_action['action.qpos.left'] = action_qpos_L.tolist()
            res_action['action.ee_pos.left'], res_action['action.ee_quat.left'] = pos_l.tolist(), rot_l.tolist()

            pos_r, rot_r = trans_mat_to_pos_quat(action_trans_R)
            res_action['action.qpos.right'] = action_qpos_R.tolist()
            res_action['action.ee_pos.right'], res_action['action.ee_quat.right'] = pos_r.tolist(), rot_r.tolist()

            if self.gripper_enable:
                action_gripper_L = action_gripper_out["left_value"]
                action_gripper_R = action_gripper_out["right_value"]
                res_action['action.gripper_qpos.left'] = action_gripper_L.tolist()
                res_action['action.gripper_qpos.right'] = action_gripper_R.tolist()
        else:
            action_qpos = action_qpos_out["value"]
            action_trans = action_trans_out["value"]
            res_action['action.qpos'] = action_qpos.tolist()

            pos, rot = trans_mat_to_pos_quat(action_trans)
            res_action['action.ee_pos'], res_action['action.ee_quat'] = pos.tolist(), rot.tolist()

            if self.gripper_enable:
                action_gripper_qpos = action_gripper_out["value"][0]
                res_action['action.gripper_qpos'] = action_gripper_qpos

        with open(robot_action_record_dir / f"{action_qpos_out['capture_time']:.3f}.json", 'w') as f:
            json.dump(res_action, f, indent=4)

    def _read_env_obs(self, rgbs, depths, trans_out, qpos_out, gripper_out):
        if self.bimanual:
            raise NotImplementedError("Bimanual real env is not implemented yet")
        elif not self.bimanual and self.robot_name == "xarm7":
            env_obs = {
                'rgbs': rgbs, # list of (H, W, 3)
                'depths': depths, # list of (H, W)
                'trans_out': trans_out["value"], # (4, 4)
                'qpos_out': qpos_out["value"], # (7,)
                'gripper_out': gripper_out["value"] # (1,)
            }

        return env_obs

    def load_actor_mlp(self,
        agent_yaml: str,
        checkpoint_path: str,
    ) -> torch.nn.Module:
        with open(agent_yaml, "r") as f:
            agent_cfg = yaml.safe_load(f)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["resume_path"] = checkpoint_path
        agent_cfg["params"]["config"]["env_info"] = {
            "observation_space": Box(-np.inf, np.inf, (35,), dtype=np.float32),
            "action_space": Box(-1.0, 1.0, (7,), dtype=np.float32),
            "agents": 1,
            "value_size": 1,
        }

        policy = PpoPlayerContinuous(agent_cfg["params"])
        policy.restore(checkpoint_path)
        policy.reset()

        return policy
    
    def get_residual_observations(self, trans_out, gripper_out, dt,
                                held_pos, fixed_pos, 
                                action_trans_out, action_gripper_out,
                                device='cuda', offline=True):
        curr_pos, fingertip_quat = trans_mat_to_pos_quat(trans_out["value"])
        gripper = gripper_raw_to_qpos(gripper_out["value"][0])
        eef_pos = torch.from_numpy(curr_pos).unsqueeze(0).to(device).to(torch.float32)
        fingertip_quat = torch.from_numpy(fingertip_quat).unsqueeze(0).to(device).to(torch.float32)
        fingertip_pos = combine_frame_transforms(
            eef_pos,
            fingertip_quat,
            torch.tensor([[0.0, 0.0, 0.23]], dtype=torch.float32).to(device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device),
        )[0]

        gripper = torch.tensor([gripper], dtype=torch.float32).unsqueeze(0).to(device)

        fixed_pos = torch.from_numpy(fixed_pos).unsqueeze(0).to(device).to(torch.float32)
        held_pos = torch.from_numpy(held_pos).unsqueeze(0).to(device).to(torch.float32)

        fingertip_pos_rel_fixed = fingertip_pos - fixed_pos  # (1,3)
        fingertip_pos_rel_held = fingertip_pos - held_pos # (1,3)

        if self.prev_fingertip_pos is None:
            ee_linvel_fd = torch.zeros((1,3), dtype=torch.float32).to(device)
            ee_angvel_fd = torch.zeros((1,3), dtype=torch.float32).to(device)
        else:
            ee_linvel_fd = (fingertip_pos - self.prev_fingertip_pos) / dt  # (1,3)
            rot_diff_quat = quat_mul(fingertip_quat, quat_conjugate(self.prev_fingertip_quat))
            rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)  # ensure shortest path
            rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
            ee_angvel_fd = rot_diff_aa / dt  # (1,3)

        self.prev_fingertip_pos = fingertip_pos
        self.prev_fingertip_quat = fingertip_quat

        if offline:
            eps_idx = torch.tensor([0], dtype=torch.int64).to(device)
            base_actions = self.base_actions_agent.get_actions(eps_idx, eef_pos, fingertip_quat, gripper)
            base_actions[:, :3] = combine_frame_transforms(
                base_actions[:, :3],
                base_actions[:, 3:7],
                torch.tensor([[0.0, 0.0, 0.23]], dtype=torch.float32).to(device),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device),
            )[0]

        else:
            base_pos, base_quat = trans_mat_to_pos_quat(action_trans_out["value"])
            base_gripper = action_gripper_out["value"][0] # NOTE: processed gripper from action commander
            base_pos = torch.from_numpy(base_pos).unsqueeze(0).to(device).to(torch.float32)
            base_quat = torch.from_numpy(base_quat).unsqueeze(0).to(device).to(torch.float32)
            base_gripper = torch.tensor([base_gripper], dtype=torch.float32).unsqueeze(0).to(device)
            base_actions = torch.cat([base_pos, base_quat, base_gripper], dim=-1)  # (1,8)
            base_actions[:, :3] = combine_frame_transforms(
                base_actions[:, :3],
                base_actions[:, 3:7],
                torch.tensor([[0.0, 0.0, 0.23]], dtype=torch.float32).to(device),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device),
            )[0]

        obs = torch.cat([
            fingertip_pos, fingertip_quat, gripper * 1.6,
            fingertip_pos_rel_fixed,
            fingertip_pos_rel_held,
            ee_linvel_fd, ee_angvel_fd,
            base_actions,
            self.prev_actions
        ], dim=-1).to(torch.float32)  # (1,35)

        return obs

    def apply_residual(self, residual, base_actions, qpos, device='cuda'):
        pos_actions = residual[:, 0:3] * torch.tensor([[0.03, 0.03, 0.03]], dtype=torch.float32).to(device)  # (1,3)
        rot_actions = residual[:, 3:6] * torch.tensor([[0.097, 0.097, 0.097]], dtype=torch.float32).to(device)  # (1,3)

        target_pos = base_actions[:, 0:3] + pos_actions  # (1,3)

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)  # (1,3)
        rot_actions_quat = quat_from_angle_axis(angle, axis)  # (1,4)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0).to(device)
        )
        target_quat = quat_mul(rot_actions_quat, base_actions[:, 3:7])  # (1,4)

        euler_xyz = torch.stack(euler_xyz_from_quat(target_quat), dim =1)
        euler_xyz[:, 0] = 3.14159
        euler_xyz[:, 1] = 0.0
        target_quat = quat_from_euler_xyz(
            roll=euler_xyz[:, 0],
            pitch=euler_xyz[:, 1],
            yaw=euler_xyz[:, 2]
        )

        target_gripper = torch.clamp(base_actions[:, 7:8], 0.0, 1.0)  # (1,1)
        # target_gripper = torch.clamp(base_actions[:, 7:8] + residual[:, 6:7], 0.0, 1.0)  # (1,1)

        actions_fingertip = torch.cat([target_pos, target_quat, target_gripper], dim=-1)  # (1,8)
        goal_fingertip_pos = actions_fingertip[:, :3].clone()
        actions_eef = actions_fingertip.clone()
        actions_eef[:, :3] = combine_frame_transforms(
            actions_eef[:, :3],
            actions_eef[:, 3:7],
            torch.tensor([[0.0, 0.0, -0.23]], dtype=torch.float32).to(device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device),
        )[0]
        actions_eef = actions_eef.squeeze(0).cpu().numpy()
        print("actions_eef: ", actions_eef)

        target_qpos = self.get_qpos_from_action_8d(actions_eef, curr_qpos=qpos)
        
        return target_qpos, goal_fingertip_pos

    def run(self) -> None:
        robot_obs_record_dir = root / "logs" / self.data_dir / self.exp_name / "robot_obs"
        os.makedirs(robot_obs_record_dir, exist_ok=True)

        robot_action_record_dir = root / "logs" / self.data_dir / self.exp_name / "robot_action"
        os.makedirs(robot_action_record_dir, exist_ok=True)

        env_obs_record_dir = root / "logs" / self.data_dir / self.exp_name / "env_obs"
        os.makedirs(env_obs_record_dir, exist_ok=True)

        # initialize kinematics helper
        self.kin_helper = KinHelper(robot_name='xarm7')

        # initialize images
        rgbs = []
        depths = []
        resolution = self.realsense.resolution
        for i in range(len(self.realsense.serial_numbers)):
            rgbs.append(np.zeros((resolution[1], resolution[0], 3), np.uint8))
            depths.append(np.zeros((resolution[1], resolution[0]), np.uint16))

        fps = self.record_fps if self.record_fps > 0 else self.realsense.capture_fps  # visualization fps
        
        timestep = 0
        eps_idx = 0
        new_eps = False
        obj_poses = {}

        if self.action_receiver == "replay" or self.action_receiver == "policy" or self.action_receiver == "residual_offline":
            if len(self.init_poses) == 1:
                init_pose = self.init_poses[0]
            else:
                init_pose = self.init_poses[eps_idx % len(self.init_poses)]
            self.set_robot_initial_pose(init_pose, fps=fps)
            print(f"Robot initial pose set for episode {eps_idx}")

            if self.action_receiver == "replay":
                total_timesteps = self.action_trajs[eps_idx].shape[0]

        self.prev_fingertip_pos = None
        self.prev_fingertip_quat = None
        get_obs_time = None
        residual = torch.zeros((1,7), dtype=torch.float32, device='cuda')

        obs_buf = []

        while self.alive:
            try:
                tic = time.time()
                state = deepcopy(self.state)

                # teleop recording
                if self.action_agent.record_start.value == True:
                    self.perception.set_record_start()
                    self.action_agent.record_start.value = False

                if self.action_agent.record_stop.value == True:
                    if self.action_agent.record_failed.value == True:
                        self.perception.set_record_failed()
                    self.perception.set_record_stop()
                    self.action_agent.record_stop.value = False

                # state data
                perception_out = state.get("perception_out", None)
                trans_out = state.get("trans_out", None)
                qpos_out = state.get("qpos_out", None)
                gripper_out = state.get("gripper_out", None)
                force_out = state.get("force_out", None)

                # action data
                action_qpos_out = state.get("action_qpos_out", None)
                action_trans_out = state.get("action_trans_out", None)
                action_gripper_out = state.get("action_gripper_out", None)

                # obj state data
                if perception_out is not None:
                    for k, v in perception_out['value'].items():
                        rgbs[k] = v["color"]
                        depths[k] = v["depth"]

                    obj_poses = self.state_estimator.estimate_object_poses( # obj2cam
                        rgbs[0],
                        depths[0],
                        retrack=self.action_agent.track_obj.value
                    )
                    if self.action_agent.track_obj.value:
                        self.action_agent.track_obj.value = False

                held2base = self.state_estimator.cam2base @ obj_poses[0]
                held_pos = held2base[:3, 3]  # obj_poses[0, :3, 3]
                fixed_pos = np.array([0.384, -0.096, 0.025]) #obj_poses[1, :3, 3]

                # reset robot to initial pose if "r" is pressed
                if self.action_agent.reset.value:
                    timestep = 0 
                    if not new_eps: # restarting at 
                        eps_idx += 1
                        new_eps = True

                        # replay terminates after all eps are done
                        if self.action_receiver == "replay":
                            if self.total_trajs <= eps_idx:
                                print("All trajectories replayed, stopping environment")
                                break
                            total_timesteps = self.action_trajs[eps_idx].shape[0]

                        print(f"Resetting environment for episode {eps_idx}")
                    
                    init_pose = self.init_poses[eps_idx % len(self.init_poses)] * np.pi/180
                    self.action_agent.command[:] = init_pose.tolist()

                # after pressing 's' to start
                elif not self.action_agent.reset.value:
                    new_eps = False
                    if self.action_receiver == "residual" or self.action_receiver == "residual_offline":
                        if get_obs_time is None:
                            dt = 1/15
                        else:
                            prev_get_obs_time = get_obs_time
                            get_obs_time = time.time()
                            dt = get_obs_time - prev_get_obs_time
                            print("dt: ", dt)

                        self.prev_actions = residual
                        obs = self.get_residual_observations(trans_out, gripper_out, dt,
                                                             held_pos, fixed_pos, 
                                                             action_trans_out, action_gripper_out,
                                                             offline=False) # tensor (1, 35)
                        obs_buf.append(obs)

                        fingertip_pos = obs[0, 0:3].cpu().numpy()
                        base_pos = obs[0, -15:-12].cpu().numpy()
                    
                        print("obs: ", obs)
                        residual = self.policy.get_action(obs, is_deterministic=True) # tensor (1,7)
                        print("residual: ", residual)
                        # pre physics step
                        residual = residual * 0.2 + self.prev_actions * 0.8  # smooth residual
                        # apply actions
                        print("base actions:", obs[:, -15:-7])  # tensor (1,8)
                        actions, final_pos = self.apply_residual(residual, obs[:, -15:-7], qpos_out["value"])  # np (8,)
                        final_pos = final_pos[0].cpu().numpy()
                        if self.action_receiver == "residual_offline":
                            self.action_agent.command[:] = actions
                            timestep += 1
                            print(f"timestep: {timestep}")
                            if timestep >= 300:
                                print(f"Episode {eps_idx} finished after {timestep} timesteps")
                                self.perception.set_record_stop()
                                torch.save(obs_buf, env_obs_record_dir / f"env_obs_eps_{eps_idx}_real.pt")
                                break

                        else:
                            self.action_agent.command_with_residual[:] = actions # np (8,)

                    elif self.action_receiver == "policy":
                        env_obs = self._read_env_obs(rgbs, depths, trans_out, qpos_out, gripper_out)

                        action = self.policy.get_action(env_obs)
                        print("current qpos:", qpos_out["value"])
                        print("qpos action:", action)
                        self.action_agent.command[:] = action

                    elif self.action_receiver == "replay":
                        if timestep == 0:
                            print(f"Starting episode {eps_idx} with {total_timesteps} timesteps")
                        if timestep < total_timesteps:
                            action = self.action_trajs[eps_idx][timestep] # (8,) in cartesian
                            # print(f"action at timestep {timestep}:", action)
                            command = self.get_qpos_from_action_8d(action, qpos_out["value"]) # (8,) in joint space
                            self.action_agent.command[:] = command
                            timestep += 1
                        else:
                            print(f"Episode {eps_idx} finished after {timestep} timesteps")
                            self.perception.set_record_stop()
                            self.action_agent.reset.value = True

                intrinsics = self.get_intrinsics()

                # TODO: compare with prev. code and debug why store data not working, mostly likely changes storing location to get_command() in action_agent?
                # store state and action data
                if trans_out is not None and action_qpos_out is not None:
                    self.store_robot_data(
                        trans_out,
                        qpos_out,
                        gripper_out,
                        action_qpos_out,
                        action_trans_out, 
                        action_gripper_out, 
                        robot_obs_record_dir,
                        robot_action_record_dir,
                        force_out,
                    )

                # Build raw full RGB+depth image (original behavior)
                row_imgs = []
                for row in range(len(self.realsense.serial_numbers)):
                    if self.state_estimator is not None and row == 0:
                        try:
                            for idx in range(obj_poses.shape[0]):
                                rgbs[row] = self.state_estimator.draw_detected_objects(
                                    rgbs[row],
                                    obj_poses[idx],
                                )
                        except BaseException as e:
                            print(f"Error in drawing detected objects")
                            pass

                        try: 
                            rgbs[row] = self.state_estimator.draw_triangle_from_base_points(
                                rgbs[row],
                                fingertip_pos,
                                base_pos,
                                final_pos
                            )
                        except BaseException as e:
                            print(f"Error in drawing triangle: {e.with_traceback()}")
                            pass

                    rgb = cv2.cvtColor(rgbs[row], cv2.COLOR_BGR2RGB)
                    depth = cv2.applyColorMap(cv2.convertScaleAbs(depths[row], alpha=0.03), cv2.COLORMAP_JET)
                    row_imgs.append(np.hstack((rgb, depth)))                
                combined_img = np.vstack(row_imgs)

                # Resize to fit window resolution set during init
                combined_img = cv2.resize(combined_img, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)

                # Write into shared memory
                np.copyto(
                    np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3)),
                    combined_img
                )

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
                if 1 / (time.time() - tic) > fps + 1 or 1 / (time.time() - tic) < fps - 1:
                    print("real env fps: ", 1 / (time.time() - tic), f"(target: {fps})")
            
            except BaseException as e:
                print(f"Error in real robot env: {e.with_traceback()}")
                break
        
        self.action_agent.stop()
        self.stop()
        print("RealEnv process stopped")

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return self.state["extr"]

    @property
    def alive(self) -> bool:
        alive = self._alive.value and self.real_alive
        self._alive.value = alive
        return alive

    def start(self) -> None:
        self.start_time = time.time()
        self._alive.value = True
        self.real_start(time.time())
        self.start_image_display()
        super().start()

    def stop(self) -> None:
        self._alive.value = False
        self.real_stop()
