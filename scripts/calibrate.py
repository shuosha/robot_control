from __future__ import annotations

from third_party.gello.robots import robot

"""
Hand–Eye calibration refactor

Final on-disk layout per session directory (work_dir):

work_dir/
 ├─ extrinsics.json   # all 4×4 transforms
 └─ intrinsics.json   # per-camera intrinsics (as dicts)

Functions (renamed + simplified IO):
  1) fixed_calibration(serials: dict[str,str])
       For each camera name in `serials`, create a RealSense manager, capture
       one RGB frame, detect Charuco, compute **board2cam** (4×4), and write:
         - extrinsics.json["board2cam"][cam_name] = ...
         - intrinsics.json[cam_name] = {...}

  2) handeye_calibration()
       Wrist-only capture/solve for: **wristcam2ee** and **ee2base**.
       Writes to extrinsics.json at top-level keys.

  3) compose()
       Pure composition, **no image capture**. Reads extrinsics.json and if it
       finds both wristcam2ee & ee2base and at least one board2cam[wrist] plus
       other board2cam[*], it computes:
         - wristcam2othercam = inv(board2cam[wrist]) @ board2cam[other]
         - othercam2base = ee2base @ wristcam2ee @ wristcam2othercam
       and stores them into extrinsics.json.

Design:
- Keep everything in a clear, minimal class. Only two JSON files are written.
- Use explicit names (board2cam, wristcam2ee, ee2base, wristcam2X, X2base).
- Charuco board and detection use OpenCV; helper conversions come from `calib`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import copy
import argparse
import glob

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

import robot_control.calibration.calibration as calib
from robot_control.calibration.ee_transformation import XArmRobot
from robot_control.calibration.realsense_manager import RealSenseManager

# ------------------
# Charuco parameters
# ------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
CALIB_BOARD = cv2.aruco.CharucoBoard(size=(6, 5), squareLength=0.04, markerLength=0.03, dictionary=ARUCO_DICT)
CHECKER_SIZE_M = 0.04
DETECTOR_PARAMS = cv2.aruco.CharucoParameters()
DETECTOR = cv2.aruco.CharucoDetector(
    CALIB_BOARD,
    DETECTOR_PARAMS,
)

# ------------------
# Wrist capture poses
# ------------------
INIT_JOINT_DEG = np.array([0, -45, 0, 30, 0, 75, 0])
PREDEFINED_JOINTS_DEG: List[List[float]] = [
    [0, -45, 0, 30, 0, 75, 0],
    [0, -25.5, 0.1, 42.6, -0.1, 68, 0],
    [3.2, -20.6, -2.7, 46.4, -1.1, 66.6, -20.4],
    [-4, -20.4, 3.7, 46.8, 1.3, 67.7, 26.6],
    [-17.7, -29.9, -12, 40.4, -6.5, 69.7, -8.7],
    [-7.4, -20.6, -17.8, 47.9, -6.5, 66.9, -43.7],
    [15.4, -23.3, 14.2, 45.6, 5.8, 68.5, 4.4],
    [4.9, -4, 19.1, 63.7, 1.9, 68.5, 59.8],
    [0, -12.3, 0.2, 54, -0.2, 66.3, 0],
    [0.1, 14.1, 0.3, 88.3, 0.5, 95.7, -0.1],
    [0.1, 19.1, 0.4, 96.1, 0.8, 109.9, 0],
    [-0.3, -50.9, 0.1, 23, -0.3, 57.5, 0],
    [-0.4, -81.1, -0.1, 7.9, -0.4, 53.8, 0],
    [0.1, -28.3, 0.1, 56.9, 0.2, 85.3, 0],
    [-0.3, -17.3, 0.4, 26.5, -0.8, 44, 0.2],
    [7.8, -21.6, 14.5, 63, 5.6, 84.6, 20.5],
    [5.5, -18.1, 21.5, 67.8, -14.4, 95.6, 25.2],
    [4.7, -3.4, 21, 81.2, -19.7, 95.3, 25.3],
    [9.1, 8.3, 17.5, 96.9, -16.1, 116.5, 18.7],
    [13.6, 11.1, 10.7, 69, -15.6, 86.3, 25],
    [15.3, -16.4, 15.3, 35.8, -25.2, 58.9, 46.3],
    [-10.4, -19.1, -17, 32.5, -6.3, 51.2, -22.9],
    [-10.5, 10.7, -11, 66.8, -2.3, 69.9, -20.9],
    [-10.7, 7.2, -10.6, 73, -2.7, 79.5, -20.9],
    [-14.9, -35.2, -19.2, 28.2, -2.6, 48.3, -31],
]

# ------------------
# Camera parameters
# ------------------
FRAME_WIDTH = 848
FRAME_HEIGHT = 480
FRAME_FPS = 30
COLOR_FRAME_FORMAT = rs.format.bgr8
DEPTH_FRAME_FORMAT = rs.format.z16

# ------------------
# Small utilities
# ------------------

def _deg_to_rad(arr_deg: np.ndarray) -> np.ndarray:
    return arr_deg / 180.0 * np.pi


def se3_log(T: np.ndarray) -> np.ndarray:
    R_mat, t = T[:3, :3], T[:3, 3]
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return np.hstack((t, rotvec))


def pose_to_matrix(t: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def handeye_residual(x: np.ndarray, T_As: List[np.ndarray], T_Bs: List[np.ndarray]) -> float:
    t, rotvec = x[:3], x[3:]
    T_A_to_B = pose_to_matrix(t, rotvec)
    total = 0.0
    for T_Ac, T_Bc in zip(T_As, T_Bs):
        T_Ac_est = T_A_to_B @ T_Bc
        delta = np.linalg.inv(T_Ac) @ T_Ac_est
        total += np.linalg.norm(se3_log(delta))
    return total / max(1, len(T_As))

def rs_intrinsics_to_dict(intr) -> dict:
    # intr: pyrealsense2.intrinsics
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),       # or int(intr.model) if you prefer numeric
        "coeffs": [float(c) for c in intr.coeffs],  # 5 coefficients
    }


def _get_images(image_path: Path) -> tuple[list[cv2.typing.MatLike], list[str]]:
    img_names = sorted(glob.glob(str(image_path)))
    images = [cv2.imread(img_name, 1) for img_name in img_names]

    return images, img_names

def _get_ee_poses(ee_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    ee_data_names = sorted(glob.glob(str(ee_path)))

    rots = []
    transls = []

    for name in ee_data_names:
        with open(name) as f:
            ee_transform = json.load(f)

        translation = np.array(ee_transform["translation"])

        # (rx, ry, rz, rw)
        quat = ee_transform["quat"]
        rot_mat = R.from_quat(quat, scalar_first=False).as_matrix()

        rots.append(rot_mat)
        transls.append(translation)

    return rots, transls, ee_data_names
# ------------------
# JSON IO helpers
# ------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))

def _pair_key(src: str, dst: str) -> str:
    return f"{src}2{dst}"

# ------------------
# Main class
# ------------------

@dataclass
class HandEyeCalibrator:
    work_dir: Path
    robot_ip: str
    show_gui: bool = True
    init_robot: bool = True

    # runtime objects
    robot: Optional[XArmRobot] = field(init=False, default=None)

    def __post_init__(self):
        self.work_dir = Path(self.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.work_dir / "calibration_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.extr_path = self.work_dir / "extrinsics.json"
        self.intr_path = self.work_dir / "intrinsics.json"

        if self.init_robot and self.robot_ip is not None:
            self.robot = XArmRobot(ip=self.robot_ip, control_frequency=500.0, max_delta=0.001)

    # ----------------------------------------------------
    # 1) fixed calibration: per-camera board2cam + intrinsics
    # ----------------------------------------------------
    def fixed_calibration(self, serials: Dict[str, str]) -> None:
        extr = _read_json(self.extr_path)
        intr = _read_json(self.intr_path)
        extr.setdefault("board2cam", {})

        for cam_name, serial in serials.items():
            cam = RealSenseManager(
                serial=serial,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=FRAME_FPS,
                depth_format=DEPTH_FRAME_FORMAT,
                color_format=COLOR_FRAME_FORMAT,
            )
            try:
                cam.poll_frames()
                bgr = cam.get_color_image()

                # intrinsics from device → OpenCV
                intr_rs = cam.get_color_intrinsic()  # dict
                intr_dict = rs_intrinsics_to_dict(intr_rs)
                mtx = np.array([[intr_dict["fx"], 0, intr_dict["ppx"]], [0, intr_dict["fy"], intr_dict["ppy"]], [0, 0, 1]])
                coeff = np.array(intr_dict["coeffs"])

                # Charuco pose → board2cam (OpenCV returns Charuco-in-cam coords)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                cc, ids, mc, mid = DETECTOR.detectBoard(gray)
                if cc is None or len(cc) < 4:
                    if self.show_gui:
                        cv2.imshow(f"Charuco not detected for camera '{cam_name}'", bgr)
                        cv2.waitKey(0)
                    raise RuntimeError(f"Charuco not detected for camera '{cam_name}'.")
                ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ids, CALIB_BOARD, mtx, np.zeros(5), None, None)
                if not ok:
                    raise RuntimeError(f"Charuco pose failed for camera '{cam_name}'.")
                R_cam, _ = cv2.Rodrigues(rvec)
                T_board2cam = calib.compose_transform(R_cam, tvec.flatten())

                # write results
                extr["board2cam"][cam_name] = np.asarray(T_board2cam).tolist()
                intr[cam_name] = intr_dict

                vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), mc, mid)
                cv2.drawFrameAxes(vis, mtx, np.zeros(5), rvec, tvec, CHECKER_SIZE_M * 2)
                cv2.imwrite(str(self.images_dir / f"{cam_name}_fixed_calibration.png"), vis)

                if self.show_gui:
                    cv2.imshow(f"{cam_name} Charuco", vis)
                    cv2.waitKey(0)
            finally:
                cam.stop()

        _write_json(self.extr_path, extr)
        _write_json(self.intr_path, intr)

    # -----------------------------------------------------------------
    # 2) hand-eye calibration: wrist-only capture → wristcam2ee & ee2base
    # -----------------------------------------------------------------
    def capture_wrist_images(self, wrist_serial) -> Tuple[np.ndarray, np.ndarray]:
        # Move to initial then iterate predefined poses
        print("Init robot")
        self.robot.move_and_wait_robot(_deg_to_rad(INIT_JOINT_DEG))

        Rees: List[np.ndarray] = []
        tees: List[np.ndarray] = []

        # Create a manager by scanning serials is out-of-scope; assume the user ran fixed_calibration and knows devices.
        # For robustness, create a RealSense without specifying serial (default to first device) if not defined.
        wrist_cam = RealSenseManager(serial=wrist_serial, width=848, height=480, fps=30, depth_format=rs.format.z16, color_format=rs.format.bgr8)

        # Grab intrinsics for pose estimation (from device now)
        wrist_cam.poll_frames()
        wrist_intr_rs = wrist_cam.get_color_intrinsic()
        wrist_intr_dict = rs_intrinsics_to_dict(wrist_intr_rs)
        wrist_mtx = np.array([[wrist_intr_dict["fx"], 0, wrist_intr_dict["ppx"]], [0, wrist_intr_dict["fy"], wrist_intr_dict["ppy"]], [0, 0, 1]])
        wrist_dist = np.array(wrist_intr_dict["coeffs"])

        for idx, joint_deg in enumerate(PREDEFINED_JOINTS_DEG):
            print(f"Capturing {idx} / {len(PREDEFINED_JOINTS_DEG) - 1}")

            self.robot.move_and_wait_robot(_deg_to_rad(np.asarray(joint_deg)))
            wrist_cam.poll_frames()
            img = copy.deepcopy(wrist_cam.get_color_image())

            # save image
            print("Saving image")
            cv2.imwrite(str(self.images_dir / f"wrist_image@pose_{idx}.png"), img)

            # save eef pose
            print("Saving End Effector Pose")
            robot_state = self.robot.get_state()
            ee_translation = robot_state.cartesian_pos().tolist()
            ee_quat = robot_state.quat().tolist()
            ee_data = {"translation": ee_translation, "quat": ee_quat}
            _write_json(self.images_dir / f"wrist_ee_pose_{idx}.json", ee_data)

            if self.show_gui:
                cv2.namedWindow(f"Wrist image {idx}", cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f"Wrist image {idx}", img)
                cv2.waitKey(100)

        print("Capture finished. Reset robot")
        self.robot.move_and_wait_robot(_deg_to_rad(INIT_JOINT_DEG))
        self.robot.stop()
        wrist_cam.stop()

    def handeye_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        extr = _read_json(self.extr_path)
        intr = _read_json(self.intr_path)

        wrist_images, img_names = _get_images(self.images_dir / "wrist_image@pose_*.png")
        Rees, tees, ee_names = _get_ee_poses(self.images_dir / "wrist_ee_pose_*.json")

        wrist_intr_dict = intr.get("wrist", {})
        wrist_mtx = np.array([[wrist_intr_dict["fx"], 0, wrist_intr_dict["ppx"]], [0, wrist_intr_dict["fy"], wrist_intr_dict["ppy"]], [0, 0, 1]])
        wrist_dist = np.array(wrist_intr_dict["coeffs"])

        # Detect Charuco for each wrist image → board2wristcam
        R_chk2cam_list, t_chk2cam_list = [] , []
        idx = 0
        for img in wrist_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cc, ids, mc, mid = DETECTOR.detectBoard(gray)
            if cc is None or len(cc) < 4:
                continue
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ids, CALIB_BOARD, wrist_mtx, np.zeros(5), None, None)
            if not ok:
                continue
            R_chk, _ = cv2.Rodrigues(rvec)
            R_chk2cam_list.append(R_chk)
            t_chk2cam_list.append(tvec)

            vis = gray.copy()[:, :, np.newaxis].repeat(3, axis=2)
            cv2.drawFrameAxes(vis, wrist_mtx, wrist_dist, rvec, tvec, 0.1)
            cv2.imwrite(str(self.images_dir / f"handeye_calibration_{idx}.png"), vis)
            if self.show_gui:
                cv2.imshow(f"Hand-Eye Calibration {idx}", vis)
                cv2.waitKey(100)

            idx += 1

        # Solve wristcam2ee using standard hand–eye
        T_wristcam2ee = calib.solve_EE_to_camera(Rees, tees, R_chk2cam_list, t_chk2cam_list)
        T_ee2base = calib.compose_transform(Rees[0], tees[0])

        # Write
        extr["wristcam2ee"] = np.asarray(T_wristcam2ee).tolist()
        extr["ee2base"] = np.asarray(T_ee2base).tolist()
        _write_json(self.extr_path, extr)

        return T_wristcam2ee, T_ee2base

    # ---------------------------------------------------------
    # 3) compose: no capture; infer wrist↔other and other→base
    # ---------------------------------------------------------
    def compose(self) -> None:
        extr = _read_json(self.extr_path)
        board2cam: dict = extr.get("board2cam", {})
        if not board2cam:
            raise RuntimeError("compose(): board2cam not found; run fixed_calibration() first.")
        if "wristcam2ee" not in extr or "ee2base" not in extr:
            raise RuntimeError("compose(): wristcam2ee or ee2base missing; run handeye_calibration() first.")

        # Identify wrist camera key heuristically: prefer key named 'wrist'
        cam_names = list(board2cam.keys())
        if "wrist" in cam_names:
            wrist_key = "wrist"
        else:
            # fallback: first key
            wrist_key = cam_names[0]

        T_wristcam2ee = np.asarray(extr["wristcam2ee"])  # 4×4
        T_ee2base = np.asarray(extr["ee2base"])          # 4×4

        T_board2wrist = np.asarray(board2cam[wrist_key])
        T_wrist2board = np.linalg.inv(T_board2wrist)

        # Prepare new containers
        extr.setdefault("cam2cam", {})
        extr.setdefault("cam2base", {})

        # Determine wrist key as you already do
        cam_names = list(board2cam.keys())
        wrist_key = "wrist" if "wrist" in cam_names else cam_names[0]

        # Pull transforms
        T_wristcam2ee = np.asarray(extr["wristcam2ee"])
        T_ee2base     = np.asarray(extr["ee2base"])
        T_board2wrist = np.asarray(board2cam[wrist_key])
        T_wrist2board = np.linalg.inv(T_board2wrist)

        # Always write wrist->base
        T_wrist2base = T_ee2base @ T_wristcam2ee  # (wristcam->ee->base)
        extr["cam2base"][_pair_key(wrist_key, "base")] = T_wrist2base.tolist()

        # For each other camera, compute wrist->other and other->base
        for other in cam_names:
            if other == wrist_key:
                continue

            T_board2other = np.asarray(board2cam[other])

            # wrist->other via board frame
            T_wrist2other = T_wrist2board @ T_board2other
            extr["cam2cam"][_pair_key(wrist_key, other)] = T_wrist2other.tolist()

            # other->base = ee->base * wristcam->ee * wrist->other
            T_other2base = T_ee2base @ T_wristcam2ee @ T_wrist2other
            extr["cam2base"][_pair_key(other, "base")] = T_other2base.tolist()

        # Persist
        _write_json(self.extr_path, extr)
        print("finish computing: wristcam2cam and cam2base for all other cameras.")

    # ------------------
    # Cleanup
    # ------------------
    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self.robot:
            self.robot.stop()

# ------------------
# Minimal example
# ------------------
if __name__ == "__main__":
    # run rs-enumerate-devices to find serials
    parser = argparse.ArgumentParser("calibration script for realsense cameras and xarm robot")
    parser.add_argument("--work-dir", required=True, help="folder to save calibration info")
    parser.add_argument("--mode", choices=["fixed", "handeye", "compose", "all"], required=True, help="calibration mode: choose from 'fixed' or 'handeye'")
    parser.add_argument("--robot-ip", default="192.168.1.196", help="IP address of the xArm robot")
    parser.add_argument("--show-gui", action="store_true", default=True, help="whether to show GUI windows during calibration")
    args = parser.parse_args()


    # Example usage (adjust IP/serials accordingly)
    work_dir = Path(args.work_dir).expanduser()
    serials = {"wrist": "130322270735", "side": "239222303153"}
    # serials = {"side": "239222303153"}

    if args.mode == "fixed":
        cal = HandEyeCalibrator(
            work_dir=work_dir,
            robot_ip=None,          
            show_gui=args.show_gui,
            init_robot=False,       
        )
        cal.fixed_calibration(serials)
    elif args.mode == "handeye":
        cal = HandEyeCalibrator(
            work_dir=work_dir, 
            robot_ip=args.robot_ip, 
            show_gui=args.show_gui
        )
        # cal.capture_wrist_images(serials["wrist"])
        cal.handeye_calibration()
    elif args.mode == "compose":
        cal = HandEyeCalibrator(
            work_dir=work_dir,
            robot_ip=None,          
            show_gui=args.show_gui,
            init_robot=False,       
        )
        cal.compose()
    elif args.mode == "all":
        cal = HandEyeCalibrator(
            work_dir=work_dir, 
            robot_ip=args.robot_ip, 
            show_gui=args.show_gui
        )
        cal.fixed_calibration(serials)
        cal.capture_wrist_images(serials["wrist"])
        cal.handeye_calibration()
        cal.compose()

    cal.close()
