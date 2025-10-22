from typing import Callable, Sequence, List, Literal, Optional
from enum import Enum
import numpy as np
import multiprocess as mp
import time
import threading
import cv2
import pygame
import os
import pickle
import transforms3d
import subprocess
from pynput import keyboard
from pathlib import Path
from copy import deepcopy
import json, glob
import torch
from scipy.spatial.transform import Rotation as R
from pprint import pprint

from meta_material.utils import get_root, mkdir
root: Path = get_root(__file__)

from modules.perception import Perception
from modules.xarm_controller import XarmController

from agents.teleop_keyboard import KeyboardTeleop
from agents.aloha_teleop import AlohaTeleop
from agents.teleop_gello import GelloTeleop

from utils.kinematics_utils import KinHelper, trans_mat_to_pos_quat, gripper_raw_to_qpos, _quat_apply, _q_mul, _q_normalize
from utils.image_transforms import InferenceTransforms

from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense

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
        exp_name: str = "recording",
        data_dir: Path = Path("data"),

        # --------------------- Cameras ---------------------
        realsense: MultiRealsense | SingleRealsense | None = None,
        shm_manager: SharedMemoryManager | None = None,
        serial_numbers: Sequence[str] | None = None,
        resolution: tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        record_fps: int | None = 0,
        record_time: float | None = 60 * 10,  # seconds
        enable_depth: bool = True,
        enable_color: bool = True,

        # --------------------- Perception ---------------------
        perception: Perception | None = None,
        perception_process_func: Callable | None = None,  # identity if None

        # --------------------- Robot ---------------------
        robot_name: Literal["xarm7", "aloha", "uf850"] = "xarm7",
        teleop_agent: Literal["gello", "keyboard", "aloha"] = "gello",
        bimanual: bool = False,
        robot_ip: Sequence[str] = ("192.168.1.220",),
        gripper_enable: bool = False,
        calibrate_result_dir: Path = Path("log/latest_calibration"),
        debug: int = 0,  # 0: silent, 1: debug, 2: verbose

        # --------------------- Control ---------------------
        input_mode: Literal["teleop", "replay", "policy"] = "teleop",
        init_pose: Sequence[float] | None = None,
        action_traj: np.ndarray | None = None,
        checkpoint_path: str | None = None,
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
            teleop_agent: Input device for teleoperation ("gello", "keyboard", or "aloha").
            bimanual: Whether to use a dual-arm setup.
            robot_ip: List of robot IP addresses (required for all modes).
            gripper_enable: Enable gripper control.
            calibrate_result_dir: Directory containing calibration results.
            debug: Debug level (0 = silent, 1 = debug, 2 = verbose).

            --------------------- Control ---------------------
            input_mode: Operation mode ("teleop", "replay", or "policy").
            init_pose: Initial robot pose.
            action_traj: Action trajectory for replay mode (T, action_dim).
            checkpoint_path: Path to policy checkpoint for policy mode.
        """
        super().__init__()

        # ------------ debug level ------------
        self.debug = 0 if debug is None else (2 if debug is True else debug)

        # ------------ logging --------------
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
                verbose=self.debug > EnvEnum.VERBOSE.value
            )
            self.serial_numbers = list(self.realsense.cameras.keys())
        
        # NOTE: hardcoded exposure and white balance for consistency
        self.realsense.set_exposure(exposure=200, gain=60, depth_exposure=10000, depth_gain=60)  # 100: bright, 60: dark
        self.realsense.set_white_balance(3800)

        # -- optional -- automatic exposure and white balance
        # self.realsense.set_exposure(exposure=None)
        # self.realsense.set_white_balance(white_balance=None)

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
                verbose=self.debug > EnvEnum.VERBOSE.value)

        # ----------- robot ---------------
        self.bimanual = bimanual

        self.init_pose = init_pose
        self.input_mode = input_mode
        self.pusht_teleop = pusht_teleop

        if input_mode == "replay":
            assert action_traj is not None, "action_traj must be provided in replay mode"
            assert init_pose is not None, "init_pose must be provided in replay mode"

            msg = "REPLAY MODE IS ACTIVATED — ACTION TRAJ IS PROVIDED"
            self.total_timesteps = action_traj.shape[0]
            self.action_traj = action_traj

        elif input_mode == "teleop":
            msg = "TELEOP MODE IS ACTIVATED — NO ACTION TRAJ PROVIDED"
        elif input_mode == "policy":
            def say(k, v): print(f"[policy] {k}: {v}")
            def banner(t): print("\n" + "="*(len(t)+10) + f"\n[policy] {t}\n" + "="*(len(t)+10))

            assert self.pusht_teleop == (task_name == "pusht"), "pusht_teleop must be consistent with task_name"
            assert task_name != 'None', "Please provide a task_name for the policy"
            self.task_name = task_name

            self.action_queue = None
            self.use_Pi0 = use_Pi0
            if self.use_Pi0:
                self.action_queue = []

                if task_name == 'insert_rope':
                    self.tf = InferenceTransforms(
                        cfg={
                            "tfs": {
                                "crop":{
                                    "kwargs": {
                                        "size": (120, 212)
                                    }
                                }
                            },
                            "downsample": {
                                "resize": {
                                    "kwargs": {
                                        "size": (120, 212)
                                    }
                                }
                            }

                        },
                        need_resize = True,
                        pusht = False
                    )
                elif task_name == 'pack_sloth':
                    self.tf = InferenceTransforms(
                        cfg={
                            "tfs": {
                                "crop":{
                                    "kwargs": {
                                        "size": (240, 240)
                                    }
                                }
                            },
                            "downsample": {
                                "resize": {
                                    "kwargs": {
                                        "size": (240, 240)
                                    }
                                }
                            }

                        },
                        need_resize = True,
                        pusht = False
                    )

                elif task_name == 'pusht':
                    self.tf = InferenceTransforms(
                        cfg={
                            "tfs": {
                                "crop":{
                                    "kwargs": {
                                        "size": (240, 240)
                                    }
                                }
                            },
                            "downsample": {
                                "resize": {
                                    "kwargs": {
                                        "size": (240, 240)
                                    }
                                }
                            }

                        },
                        need_resize = True,
                        pusht = True
                    )
                else:
                    raise NotImplementedError(f"Unsupported task_name for Pi0: {task_name}")

                self.rel_actions = False # Pi0 handles relative actions internally; output always abs

                self.obs_dict = {
                    "observation.images.front": None,
                    "observation.images.wrist": None,
                    "observation.state": None,
                    # "task": None
                }

                # insert rope
                if task_name == 'insert_rope':
                    config = pi0_cfg.get_config("pi0_lora_insert_rope") # NOTE default relative
                elif task_name == 'pack_sloth':
                    config = pi0_cfg.get_config("pi0_lora_pack_sloth_rel") # NOTE default relative
                elif task_name == 'pusht':
                    config = pi0_cfg.get_config("pi0_lora_pusht_rel")

                checkpoint_dir = download.maybe_download(checkpoint_path)
                
                self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

            else:
                ckpt = Path(checkpoint_path)
                cfg_path = ckpt / "train_config.json"
                with open(cfg_path) as f: 
                    cfg = json.load(f)

                self.tf = InferenceTransforms(cfg=cfg["dataset"]["image_transforms"], need_resize=True, pusht=self.pusht_teleop)
                ptype = cfg["policy"]["type"]
                self.rel_actions = bool(cfg["policy"].get("relative_actions", False))
                repo_id = cfg["dataset"].get("repo_id")

                cls_map = {"diffusion": DiffusionPolicy, "act": ACTPolicy, "pi0": PI0Policy, "smolvla": SmolVLAPolicy}
                if ptype not in cls_map: 
                    raise NotImplementedError(f"Unsupported policy type: {ptype}")

                rel_stats = None
                if self.rel_actions and repo_id:
                    meta = Path.home() / ".cache/huggingface/lerobot" / repo_id / "meta"
                    if ptype == "diffusion":
                        cand = [meta / "relative_action_stats_Te64.pt"]
                    else:
                        cand = [meta / "relative_action_stats_Te50.pt"]
                    rel_stats = torch.load(cand[0]); say("relative_action_stats", cand[0])

                Policy = cls_map[ptype]
                self.policy = Policy.from_pretrained(str(ckpt), **({"rel_action_stats": rel_stats} if rel_stats is not None else {}))

                inp, out = self.policy.config.input_features, self.policy.config.output_features
                banner("loaded")
                say("type", ptype)
                say("checkpoint", ckpt)
                say("relative_actions", self.rel_actions)
                say("inputs", inp)
                say("outputs", out)

                # persist brief info
                info_dir = root / "log" / self.data_dir / self.exp_name / "infos"
                mkdir(info_dir, overwrite=False, resume=False)
                with open(info_dir / "policy_info.json", "w") as f:
                    json.dump({
                        "policy_type": ptype,
                        "checkpoint_path": str(ckpt),
                        "relative_actions": self.rel_actions,
                        "repo_id": repo_id,
                        "input_features": {k: v.__dict__ for k, v in inp.items()},
                        "output_features": {k: v.__dict__ for k, v in out.items()},
                    }, f, indent=2)

                self.obs_dict = {k: None for k in inp.keys()}


        # # base calibration
        # self.calibrate_result_dir = calibrate_result_dir
        # with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
        #     base = pickle.load(f)
        # if self.bimanual and use_xarm:
        #     R_leftbase2board = base['R_leftbase2world']
        #     t_leftbase2board = base['t_leftbase2world']
        #     R_rightbase2board = base['R_rightbase2world']
        #     t_rightbase2board = base['t_rightbase2world']
        #     leftbase2world_mat = np.eye(4)
        #     leftbase2world_mat[:3, :3] = R_leftbase2board
        #     leftbase2world_mat[:3, 3] = t_leftbase2board
        #     self.state["b2w_l"] = leftbase2world_mat
        #     rightbase2world_mat = np.eye(4)
        #     rightbase2world_mat[:3, :3] = R_rightbase2board
        #     rightbase2world_mat[:3, 3] = t_rightbase2board
        #     self.state["b2w_r"] = rightbase2world_mat
        # else:
        #     R_base2board = base['R_base2world']
        #     t_base2board = base['t_base2world']
        #     base2world_mat = np.eye(4)
        #     base2world_mat[:3, :3] = R_base2board
        #     base2world_mat[:3, 3] = t_base2board
        #     self.state["b2w"] = base2world_mat

        # # camera calibration
        # extr_list = []
        # with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
        #     rvecs = pickle.load(f)
        # with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
        #     tvecs = pickle.load(f)
        # for i in range(len(self.serial_numbers)):
        #     device = self.serial_numbers[i]
        #     if device not in rvecs:
        #         print(f"Device {device} not found in rvecs, skipping...")
        #         device = list(rvecs.keys())[0]
        #     R_world2cam = cv2.Rodrigues(rvecs[device])[0]
        #     t_world2cam = tvecs[device][:, 0]
        #     extr_mat = np.eye(4)
        #     extr_mat[:3, :3] = R_world2cam
        #     extr_mat[:3, 3] = t_world2cam
        #     extr_list.append(extr_mat)
        # self.state["extr"] = np.stack(extr_list)

        # # save calibration
        # mkdir(root / "log" / self.data_dir / self.exp_name / "calibration", overwrite=False, resume=False)
        # subprocess.run(f'cp -r {self.calibrate_result_dir}/* {str(root)}/log/{self.data_dir}/{self.exp_name}/calibration', shell=True)

        # Robot setup
        assert not (use_xarm and use_aloha), "Cannot use both xArm and Aloha robot at the same time"
        self.use_gello = use_gello
        self.use_aloha = use_aloha
        self.use_xarm = use_xarm

        if self.use_aloha:
            self.bimanual = True # aloha is always bimanual (for now)

        if use_xarm:
            if self.bimanual:
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip[0],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=0,
                    verbose=False,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip[1],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=1,
                    verbose=False,
                )
                self.xarm_controller = None
            else:
                self.xarm_controller = XarmController( # TODO: add admittance control mode
                    start_time=time.time(),
                    ip=robot_ip,
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if (use_gello or pusht_teleop) else 'cartesian',
                    robot_id=-1,
                    verbose=False,
                )
                self.left_xarm_controller = None
                self.right_xarm_controller = None
        else:
            self.left_xarm_controller = None
            self.right_xarm_controller = None
            self.xarm_controller = None

        self.state = mp.Manager().dict()  # should be main explict exposed variable to the child class / process

        # subprocess can only start a process object created by current process
        self._real_alive = mp.Value('b', False)

        self.start_time = 0
        mp.Process.__init__(self)
        self._alive = mp.Value('b', False)

        # command input setup: Teleop/replay/policy
        if self.use_xarm:
            if self.use_gello and gripper_enable:
                self.teleop = GelloTeleop(bimanual=self.bimanual, teleop_mode=True if input_mode == "teleop" else False)
            elif self.pusht_teleop and not gripper_enable:
                self.teleop = GelloTeleop(bimanual=self.bimanual, teleop_mode=True if input_mode == "teleop" else False, pusht_mode=True, gripper_enable=False)
                # self.teleop = PushtTeleop(teleop_mode=True if input_mode == "teleop" else False)
            else:
                self.teleop = KeyboardTeleop()
        elif self.use_aloha:
            self.teleop = AlohaTeleop()
        else:
            raise NotImplementedError("Only xArm and Aloha robots are supported for teleoperation")

        # pygame
        # Initialize a separate Pygame window for image display
        # Define image shape from single RGB+depth view
        img_w, img_h = 848, 480
        views_per_cam = 2  # RGB + depth
        num_cams = len(self.realsense.serial_numbers)

        # Image grid: stacked vertically
        unscaled_width  = img_w * views_per_cam  # 2 * 848 = 1696
        unscaled_height = img_h * num_cams       # e.g., 480 * 4 = 1920

        # Get max screen size
        pygame.init()
        screen_info = pygame.display.Info()
        max_screen_w, max_screen_h = screen_info.current_w, screen_info.current_h

        # Optional: leave a margin so the window title bar doesn't overflow
        max_screen_h = int(max_screen_h * 0.95)

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

        # record robot action
        # self.robot_record_restart = mp.Value('b', False)
        # self.robot_record_stop = mp.Value('b', False)

        # robot eef
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame
        self.gripper_enable = gripper_enable
        
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
        if self.use_xarm:
            if self.bimanual:
                self.left_xarm_controller.start()
                self.right_xarm_controller.start()
            else:
                self.xarm_controller.start()

        if self.use_gello or self.use_aloha or self.pusht_teleop:
            self.teleop.start()
        
        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)
        
        # get intrinsics
        # intrs = self.get_intrinsics()
        # intrs = np.array(intrs)
        # np.save(root / "log" / self.data_dir / self.exp_name / "calibration" / "intrinsics.npy", intrs)
        
        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        self._real_alive.value = False
        if self.use_xarm:
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
        print("real env stopped")

    @property
    def real_alive(self) -> bool:
        alive = self._real_alive.value
        if self.perception is not None:
            alive = alive and self.perception.alive.value
        if self.use_xarm:
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
        if self.bimanual and self.use_xarm:
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
        elif self.use_xarm:
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
        elif self.use_aloha:
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
        if self.bimanual and self.use_xarm:
            raise NotImplementedError("Bimanual command update is not implemented yet")
        elif self.use_xarm and self.use_gello and not self.pusht_teleop:
            if self.teleop.is_alive():
                self.state["action_qpos_out"] = {
                    "capture_time": self.teleop.cur_time_q.value,
                    "value": np.array(self.teleop.cur_comm_qpos_q[:])
                }
                self.state["action_trans_out"] = {
                    "value": np.array(self.teleop.cur_comm_trans_q[:]).reshape(4, 4)
                }
                self.state["action_gripper_out"] = {
                    "value": np.array(self.teleop.cur_comm_gripper_q[:])
                }
        elif self.use_xarm and self.pusht_teleop:
            if self.teleop.is_alive():
                self.state["action_xy_out"] = {
                    "value": np.array(self.teleop.cur_comm_xy_q[:])
                }
                self.state["action_qpos_out"] = {
                    "capture_time": self.teleop.cur_time_q.value,
                    "value": np.array(self.teleop.cur_comm_qpos_q[:])
                }
        elif self.use_aloha:
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
                if self.use_xarm or self.use_aloha:
                    self._update_robot()
                if self.perception is not None:
                    self._update_perception()
                if (self.use_gello and self.use_xarm) or self.use_aloha or self.pusht_teleop:
                    self._update_command()
            except BaseException as e:
                print(f"Error in update_real_state: {e.with_traceback()}")
                break
        print("update_real_state stopped")

    def display_image(self):
        pygame.init()
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

    def run(self) -> None:
        robot_obs_record_dir = root / "log" / self.data_dir / self.exp_name / "robot_obs"
        os.makedirs(robot_obs_record_dir, exist_ok=True)

        robot_action_record_dir = root / "log" / self.data_dir / self.exp_name / "robot_action"
        os.makedirs(robot_action_record_dir, exist_ok=True)

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

        if self.input_mode == "replay":
            print("Resetting robot to initial pose:", self.init_pose)
            init_qpos = self.get_qpos_from_action_8d(self.init_pose, np.array([0.0, -0.79, 0.0, 0.52, 0.0, 1.31, 0.0]))  # convert init pose to qpos
            
            if self.bimanual:
                raise NotImplementedError("Bimanual replay is not implemented yet")
            else:
                if self.pusht_teleop:
                    init_qpos = init_qpos[:7]
                while self.alive:
                    self.xarm_controller.teleop_activated.value = True
                    for _ in range(100):
                        tic = time.time()
                        self.teleop.command[:] = init_qpos
                        time.sleep(max(0, 1 / fps - (time.time() - tic)))
                    time.sleep(1)
                    self.teleop.record_start.value = True
                    break
        
        match_sim = True
        if self.input_mode == "policy":
            if self.bimanual:
                raise NotImplementedError("Bimanual policy is not implemented yet")
            else:
                if self.pusht_teleop:
                    # INIT_QPOS = np.array([18.5, 25.3, 4.1, 69.8, -2.9, 45, 23.6]) / 180 * np.pi  # pusht eval initial pose
                    INIT_QPOS = np.array([-16.6, -4.5, -17.1, 28.6, -3.7, 32.3, -31]) / 180 * np.pi  # pusht orange dot
                elif match_sim:
                    sim_init_pose = np.array([0.25679999589920044, 0.0, 0.40049999952316284, 0.0, 1.0, 0.0, 0.0, 0.0])
                    INIT_QPOS = self.get_qpos_from_action_8d(sim_init_pose, np.array([0.0, -0.79, 0.0, 0.52, 0.0, 1.31, 0.0]))
                else:
                    INIT_QPOS = np.array([0.0, -0.79, 0.0, 0.52, 0.0, 1.31, 0.0, 0.0]) # default pose
                self.xarm_controller.teleop_activated.value = True
                for _ in range(50):
                        tic = time.time()
                        self.teleop.command[:] = INIT_QPOS
                        time.sleep(max(0, 1 / fps - (time.time() - tic)))
                self.teleop.record_start.value = True

        chunk_size = 50        
        idx = 0
        while self.alive:
            if self.teleop.initialize_done.value:
                try:
                    tic = time.time()
                    state = deepcopy(self.state)

                    if self.teleop.record_start.value == True:
                        self.perception.set_record_start()
                        self.teleop.record_start.value = False

                    if self.teleop.record_stop.value == True:
                        if self.teleop.record_failed.value == True:
                            self.perception.set_record_failed()
                        self.perception.set_record_stop()
                        self.teleop.record_stop.value = False

                    # state data
                    perception_out = state.get("perception_out", None)
                    trans_out = state.get("trans_out", None)
                    qpos_out = state.get("qpos_out", None)
                    gripper_out = state.get("gripper_out", None)

                    # # action data
                    action_qpos_out = state.get("action_qpos_out", None)
                    if self.pusht_teleop:
                        action_xy_out = state.get("action_xy_out", None)
                    else:
                        action_trans_out = state.get("action_trans_out", None)
                        action_gripper_out = state.get("action_gripper_out", None)

                    # TODO: like gello, move policy & replay to a separate thread
                    # replay traj
                    if self.input_mode == "replay":
                        if self.bimanual:
                            raise NotImplementedError("Bimanual replay is not implemented yet")
                        else:
                            if idx < self.total_timesteps:
                                print(f"Step {idx}/{self.total_timesteps}, action: {self.action_traj[idx]}")
                                if self.pusht_teleop:
                                    pos_xy = self.action_traj[idx][:2]
                                    pos_z = 0.22
                                    r, p, y = 180.0 / 180 * np.pi, 0.0, 0.0
                                    cartesian = np.concatenate([pos_xy, [pos_z, r, p, y]], axis=0)  # (6,)
                                    qpos_goal = self.kin_helper.compute_ik_sapien(initial_qpos=qpos_out["value"], cartesian=cartesian)
                                else:
                                    qpos_goal = self.get_qpos_from_action_8d(self.action_traj[idx], qpos_out["value"])
                                print("qpos_goal:", qpos_goal)
                                self.teleop.command[:] = qpos_goal
                                idx += 1
                            else:
                                print("End of action trajectory, stopping robot control")
                                break

                    # rollout policy
                    elif self.input_mode == "policy":
                        if perception_out is not None:
                            if self.bimanual:
                                if "observation.state" in self.obs_dict:
                                    ee2base_L = deepcopy(trans_out["right_value"])
                                    ee2base_R = deepcopy(trans_out["left_value"])
                                    pos_l = ee2base_L[:3, 3]  # end effector position in base frame
                                    pos_r = ee2base_R[:3, 3]  # end effector position in base frame

                                    quat_xyzw_l = R.from_matrix(ee2base_L[:3, :3]).as_quat()  # end effector orientation in base frame
                                    quat_wxyz_l = np.roll(quat_xyzw_l, 1)
                                    gripper_raw_l = np.clip(float(gripper_out["left_value"]), 0.0, 800.0)  # gripper position
                                    gripper_qpos_l = (800.0 - gripper_raw_l) / 800.0  # normalize to [0, 1]
                                    state_np_l = np.concatenate([pos_l, quat_wxyz_l, [gripper_qpos_l]], axis=0)  # (8,)

                                    quat_xyzw_r = R.from_matrix(ee2base_R[:3, :3]).as_quat()  # end effector orientation in base frame
                                    quat_wxyz_r = np.roll(quat_xyzw_r, 1)
                                    gripper_raw_r = np.clip(float(gripper_out["right_value"]), 0.0, 800.0)  # gripper position
                                    gripper_qpos_r = (800.0 - gripper_raw_r) / 800.0  # normalize to [0, 1]
                                    state_np_r = np.concatenate([pos_r, quat_wxyz_r, [gripper_qpos_r]], axis=0)  # (8,)

                                    state_np = np.concatenate([state_np_r, state_np_l], axis=0)  # (16,)
                                    self.obs_dict["observation.state"] = torch.from_numpy(state_np.reshape(1,-1)).float().to(device='cuda:0')  # (1, 16)

                                if "observation.images.front" in self.obs_dict:
                                    front_image_np = deepcopy(perception_out['value'][0]["color"]).astype(np.float32) / 255.0
                                    # # bgr to rgb
                                    front_image_np = front_image_np[..., ::-1].copy()
                                    front_image_tensor = torch.from_numpy(front_image_np).permute(2,0,1).unsqueeze(0).to(device='cuda:0')
                                    if hasattr(self, "tf"):
                                        front_image_tensor = self.tf(
                                            front_image_tensor, 
                                            pack_sloth=True if self.task_name == "pack_sloth" else False,
                                            overlay=True if self.pusht_teleop else False
                                        )

                                    self.obs_dict["observation.images.front"] = front_image_tensor

                                if "observation.images.left" in self.obs_dict:
                                    left_image_np = deepcopy(perception_out['value'][1]["color"]).astype(np.float32) / 255.0
                                    # # bgr to rgb
                                    left_image_np = left_image_np[..., ::-1].copy()
                                    left_image_tensor = torch.from_numpy(left_image_np).permute(2,0,1).unsqueeze(0).to(device='cuda:0') # (1, C, H, W)
                                    if hasattr(self, "tf"):
                                        left_image_tensor = self.tf(
                                            left_image_tensor, 
                                            pack_sloth=True if self.task_name == "pack_sloth" else False, 
                                            overlay=False
                                        )
                                    self.obs_dict["observation.images.left"] = left_image_tensor

                                if "observation.images.right" in self.obs_dict:
                                    right_image_np = deepcopy(perception_out['value'][2]["color"]).astype(np.float32) / 255.0
                                    # # bgr to rgb
                                    right_image_np = right_image_np[..., ::-1].copy()
                                    right_image_tensor = torch.from_numpy(right_image_np).permute(2,0,1).unsqueeze(0).to(device='cuda:0') # (1, C, H, W)
                                    if hasattr(self, "tf"):
                                        right_image_tensor = self.tf(
                                            right_image_tensor, 
                                            pack_sloth=True if self.task_name == "pack_sloth" else False, 
                                            overlay=False
                                        )
                                    self.obs_dict["observation.images.right"] = right_image_tensor

                                if "task" not in self.obs_dict:
                                    B = self.obs_dict["observation.state"].shape[0]    # number of parallel envs
                                    self.obs_dict["task"] = [self.task_name] * B 
                            else:
                                if "observation.state" in self.obs_dict:
                                    ee2base = deepcopy(trans_out["value"])
                                    pos = ee2base[:3, 3]  # end effector position in base frame

                                    if self.pusht_teleop:
                                        pos_xy = pos[:2]  # end effector position in base frame (x, y)
                                        self.obs_dict["observation.state"] = torch.from_numpy(pos_xy.reshape(1, -1)).float().to(device='cuda:0')  # (1, 2)
                                    else:
                                        quat_xyzw = R.from_matrix(ee2base[:3, :3]).as_quat()  # end effector orientation in base frame
                                        quat_wxyz = np.roll(quat_xyzw, 1)
                                        gripper_raw = np.clip(float(gripper_out["value"]), 0.0, 800.0)  # gripper position
                                        gripper_qpos = (800.0 - gripper_raw) / 800.0  # normalize to [0, 1]
                                        state_np = np.concatenate([pos, quat_wxyz, [gripper_qpos]], axis=0)  # (8,)
                                        self.obs_dict["observation.state"] = torch.from_numpy(state_np.reshape(1,-1)).float().to(device='cuda:0')  # (1, 8)

                                if "observation.images.front" in self.obs_dict:
                                    front_image_np = deepcopy(perception_out['value'][0]["color"]).astype(np.float32) / 255.0
                                    # # bgr to rgb
                                    front_image_np = front_image_np[..., ::-1].copy()
                                    front_image_tensor = torch.from_numpy(front_image_np).permute(2,0,1).unsqueeze(0).to(device='cuda:0')
                                    if hasattr(self, "tf"):
                                        front_image_tensor = self.tf(
                                            front_image_tensor, 
                                            pack_sloth=True if self.task_name == "pack_sloth" else False,
                                            overlay=True if self.pusht_teleop else False
                                        )

                                    self.obs_dict["observation.images.front"] = front_image_tensor

                                if "observation.images.wrist" in self.obs_dict:
                                    wrist_image_np = deepcopy(perception_out['value'][1]["color"]).astype(np.float32) / 255.0
                                    # # bgr to rgb
                                    wrist_image_np = wrist_image_np[..., ::-1].copy()
                                    wrist_image_tensor = torch.from_numpy(wrist_image_np).permute(2,0,1).unsqueeze(0).to(device='cuda:0') # (1, C, H, W)
                                    if hasattr(self, "tf"):
                                        wrist_image_tensor = self.tf(
                                            wrist_image_tensor, 
                                            pack_sloth=True if self.task_name == "pack_sloth" else False, 
                                            overlay=False
                                        )
                                    self.obs_dict["observation.images.wrist"] = wrist_image_tensor

                                if "task" not in self.obs_dict:
                                    B = self.obs_dict["observation.state"].shape[0]    # number of parallel envs
                                    self.obs_dict["task"] = [self.task_name] * B 
                                
                        if not self.teleop.reset.value:
                            if self.use_Pi0:
                                assert self.action_queue is not None
                                for k, v in self.obs_dict.items():
                                    if k != "task":
                                        v_new = v.detach().clone()[0]
                                        if k != "observation.state":
                                            v_new = v_new.permute(1, 2, 0)
                                        v_new = v_new.cpu().numpy()
                                    else:
                                        v_new = v[0]
                                    self.obs_dict[k] = v_new

                                # print("obs shapes:", {k: v.shape for k, v in self.obs_dict.items() if k != "task"})

                                if len(self.action_queue) == 0:
                                    # print('#####################################')
                                    policy_output = self.policy.infer(self.obs_dict)
                                    # print(policy_output.keys())
                                    action_chunk = policy_output["action"]
                                    # print(action_chunk.shape)
                                    # print("action_chunk:", action_chunk)
                                    action_chunk = torch.from_numpy(action_chunk).float().to(device='cuda:0')

                                    self.action_queue = [action_chunk[i] for i in range(action_chunk.shape[0])]
                                    # print('#####################################')

                                action = self.action_queue.pop(0).unsqueeze(0)  # (1, 8)
                            else:
                                with torch.no_grad():
                                    action = self.policy.select_action(self.obs_dict) # (1,8)

                            if self.pusht_teleop:
                                if self.rel_actions:
                                    if idx % chunk_size == 0:
                                        p0 = self.obs_dict["observation.state"][:,:2].clone()

                                    pos_xy = p0 + action[:,:2]
                                    pos_xy = pos_xy.reshape(-1).cpu().numpy()  # relative position (x, y)
                                else:
                                    pos_xy = action.reshape(-1).cpu().numpy() #(2, )
                                pos_z = 0.22
                                r, p, y = -180 / 180 * np.pi, 0.0, 0.0
                                cartesian = np.concatenate([pos_xy, [pos_z, r, p, y]], axis=0)  # (6,)
                                qpos_goal = self.kin_helper.compute_ik_sapien(initial_qpos=qpos_out["value"], cartesian=cartesian)

                            else:
                                if self.rel_actions:
                                    if idx % chunk_size == 0:
                                        p0 = self.obs_dict["observation.state"][:,:3].clone()
                                        q0 = _q_normalize(self.obs_dict["observation.state"][:,3:7].clone())
                                        g0 = self.obs_dict["observation.state"][:,-1:].clone()

                                    pt = p0 + _quat_apply(q0, action[:,:3])  # relative position
                                    qt = _q_mul(q0, _q_normalize(action[:,3:7]))  # relative orientation
                                    gt = action[:,-1:]  # relative gripper position
                                    action = torch.cat([pt, qt, gt], dim=-1)  # (1, 8)

                                action[:,2] = torch.clamp(action[:,2], min=0.175)  # clamp z position
                                qpos_goal =  self.get_qpos_from_action_8d(action.reshape(-1), qpos_out["value"])

                            print(f"Step {idx}, curr state: \n {self.obs_dict['observation.state']}, \n") 
                            print("action_goal:", action)

                            self.teleop.command[:] = qpos_goal
                            idx += 1

                        elif self.teleop.reset.value:
                            self.teleop.command[:] = INIT_QPOS

                            if self.use_Pi0:
                                self.action_queue = []

                            self.policy.reset() # reset policy buffers
                            idx = 0
                            
                    intrinsics = self.get_intrinsics()
                    if perception_out is not None:
                        for k, v in perception_out['value'].items():
                            rgbs[k] = v["color"]
                            depths[k] = v["depth"]
                            intr = intrinsics[k]

                            l = 0.1
                            origin = state["extr"][k] @ np.array([0, 0, 0, 1])
                            x_axis = state["extr"][k] @ np.array([l, 0, 0, 1])
                            y_axis = state["extr"][k] @ np.array([0, l, 0, 1])
                            z_axis = state["extr"][k] @ np.array([0, 0, l, 1])
                            origin = origin[:3] / origin[2]
                            x_axis = x_axis[:3] / x_axis[2]
                            y_axis = y_axis[:3] / y_axis[2]
                            z_axis = z_axis[:3] / z_axis[2]
                            origin = intr @ origin
                            x_axis = intr @ x_axis
                            y_axis = intr @ y_axis
                            z_axis = intr @ z_axis
                            cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(x_axis[0]), int(x_axis[1])), (255, 0, 0), 2)
                            cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(y_axis[0]), int(y_axis[1])), (0, 255, 0), 2)
                            cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(z_axis[0]), int(z_axis[1])), (0, 0, 255), 2)

                            eef_points = np.concatenate([self.eef_point, np.ones((self.eef_point.shape[0], 1))], axis=1)  # (n, 4)
                            eef_colors = [(0, 255, 255)]
                            eef_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # (3, 4)
                            eef_axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

                            # CLEAN UP THE LOGIC
                            if (self.use_xarm or self.use_aloha) and trans_out is not None and not self.pusht_teleop:
                                res = {}
                                if self.bimanual:
                                    for val, prefix in zip(["left_value", "right_value"], ["left", "right"]):
                                        ee2base = trans_out[val]  # (4, 4)
                                        pos, quat = trans_mat_to_pos_quat(ee2base)
                                        res[f"obs.ee_pos.{prefix}"], res[f"obs.ee_quat.{prefix}"] = pos.tolist(), quat.tolist()
                                        # res[f"obs.ee2base.{prefix}"] = ee2base.tolist()  # (4, 4)
                                        # eef_points_world = (ee2base @ eef_points.T).T[:, :3]  # (n, 3)
                                        # eef_orientation_world = ee2base[:3, :3] @ eef_axis[:, :3].T  # (3, 3)
                                        # eef2b = np.eye(4)
                                        # eef2b[:3, :3] = eef_orientation_world
                                        # eef2b[:3, 3] = eef_points_world[0]

                                        # res[f"{prefix}_e2b"] = eef2b.tolist()  # (4, 4)

                                else:
                                    ee2base = trans_out["value"]  # (4, 4)
                                    pos, quat = trans_mat_to_pos_quat(ee2base)
                                    res["obs.ee_pos"], res["obs.ee_quat"] = pos.tolist(), quat.tolist()

                                # add gripper
                                if self.gripper_enable:
                                    if self.bimanual: # NOTE: check raw values to see if gripper_raw_to_qpos is needed
                                        left_gripper = gripper_out["left_value"]
                                        right_gripper = gripper_out["right_value"]
                                        res['obs.gripper_qpos.left'] = left_gripper.tolist() # edited to _qpos
                                        res['obs.gripper_qpos.right'] = right_gripper.tolist()
                                    else:
                                        gripper = gripper_out["value"][0]
                                        res['obs.gripper_qpos'] = gripper_raw_to_qpos(gripper) # NOTE: not np array for xarm

                                # add qpos
                                if self.bimanual:
                                    left_qpos = qpos_out["left_value"]
                                    right_qpos = qpos_out["right_value"]
                                    res['obs.qpos.left'] = left_qpos.tolist()
                                    res['obs.qpos.right'] = right_qpos.tolist()
                                else:
                                    qpos = qpos_out["value"]
                                    res['obs.qpos'] = qpos.tolist()

                                with open(robot_obs_record_dir / f"{trans_out['capture_time']:.3f}.json", 'w') as f:
                                    json.dump(res, f, indent=4)
                                
                                # save action in a different file to avoid timestamp conflicts
                                res_action = {}

                                # add action
                                if self.bimanual:
                                    action_qpos_L = action_qpos_out["left_value"]
                                    action_trans_L = action_trans_out["left_value"]
                                    action_gripper_L = action_gripper_out["left_value"]

                                    action_qpos_R = action_qpos_out["right_value"]
                                    action_trans_R = action_trans_out["right_value"]
                                    action_gripper_R = action_gripper_out["right_value"]

                                    pos_l, rot_l = trans_mat_to_pos_quat(action_trans_L)
                                    res_action['action.qpos.left'] = action_qpos_L.tolist()
                                    res_action['action.ee_pos.left'], res_action['action.ee_quat.left'] = pos_l.tolist(), rot_l.tolist()
                                    # res_action['action.cartesian.left'] = action_trans_L.tolist()
                                    res_action['action.gripper_qpos.left'] = action_gripper_L.tolist()

                                    pos_r, rot_r = trans_mat_to_pos_quat(action_trans_R)
                                    res_action['action.qpos.right'] = action_qpos_R.tolist()
                                    res_action['action.ee_pos.right'], res_action['action.ee_quat.right'] = pos_r.tolist(), rot_r.tolist()
                                    # res_action['action.cartesian.right'] = action_trans_R.tolist()
                                    res_action['action.gripper_qpos.right'] = action_gripper_R.tolist()
                                else:
                                    action_qpos = action_qpos_out["value"]
                                    action_trans = action_trans_out["value"]
                                    action_gripper_qpos = action_gripper_out["value"][0]
                                    res_action['action.qpos'] = action_qpos.tolist()

                                    pos, rot = trans_mat_to_pos_quat(action_trans)
                                    res_action['action.ee_pos'], res_action['action.ee_quat'] = pos.tolist(), rot.tolist()
                                    res_action['action.gripper_qpos'] = action_gripper_qpos

                                with open(robot_action_record_dir / f"{action_qpos_out['capture_time']:.3f}.json", 'w') as f:
                                    json.dump(res_action, f, indent=4)

                            elif self.pusht_teleop:
                                res = {}

                                # obs
                                ee2base = trans_out["value"]  # (4, 4)
                                pos, quat = trans_mat_to_pos_quat(ee2base)
                                res["obs.xy"] = pos[:2].tolist()

                                # still add full pos and quat for conveniences
                                res["obs.ee_pos"], res["obs.ee_quat"] = pos.tolist(), quat.tolist()

                                qpos = qpos_out["value"]
                                res['obs.qpos'] = qpos.tolist()

                                with open(robot_obs_record_dir / f"{trans_out['capture_time']:.3f}.json", 'w') as f:
                                    json.dump(res, f, indent=4)

                                res_action = {}

                                # action
                                if action_qpos_out is not None:
                                    action_qpos = action_qpos_out["value"]
                                    res_action['action.qpos'] = action_qpos.tolist()

                                if action_xy_out is not None:
                                    action_xy = action_xy_out["value"]
                                    res_action['action.xy'] = action_xy.tolist()

                                with open(robot_action_record_dir / f"{action_qpos_out['capture_time']:.3f}.json", 'w') as f:
                                    json.dump(res_action, f, indent=4)


                    if self.input_mode == "policy":
                        # visualize actual policy inputs (already RGB in [0,1] tensors)
                        def to_vis(t):
                            if isinstance(t, np.ndarray):
                                return (t * 255).astype(np.uint8)
                            t = t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # (H,W,C) RGB [0,1]
                            return (t * 255).astype(np.uint8)

                        vis_imgs = []

                        if self.bimanual:
                            img_front = self.obs_dict.get("observation.images.front")
                            img_left = self.obs_dict.get("observation.images.left")
                            img_right = self.obs_dict.get("observation.images.right")
                            if img_front is not None:
                                vis_imgs.append(to_vis(img_front))
                            if img_left is not None:
                                vis_imgs.append(to_vis(img_left))
                            if img_right is not None:
                                vis_imgs.append(to_vis(img_right))
                        else:
                            img_front = self.obs_dict.get("observation.images.front")
                            img_wrist = self.obs_dict.get("observation.images.wrist")
                            if img_front is not None:
                                vis_imgs.append(to_vis(img_front))
                            if img_wrist is not None:
                                vis_imgs.append(to_vis(img_wrist))

                        policy_viz = np.vstack(vis_imgs)
                        policy_viz = cv2.resize(policy_viz, (self.screen_width // 2, self.screen_height), interpolation=cv2.INTER_AREA)
                        realsense_viz = np.vstack([cv2.cvtColor(rgbs[0], cv2.COLOR_BGR2RGB), cv2.cvtColor(rgbs[1], cv2.COLOR_BGR2RGB)]) # may need to edit this line for bimanual

                        all_imgs = [policy_viz, realsense_viz]

                        if vis_imgs:
                            # horizontally stack policy inputs; they should share the same H from your tf
                            combined_img = np.hstack(all_imgs)
                        else:
                            # fallback if no images (shouldn't happen)
                            combined_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

                    else:
                        # Build raw full RGB+depth image (original behavior)
                        row_imgs = []
                        for row in range(len(self.realsense.serial_numbers)):
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
                
                except BaseException as e:
                    print(f"Error in real robot env: {e.with_traceback()}")
                    break
        
        if self.use_xarm or self.use_aloha:
            self.teleop.stop()
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
