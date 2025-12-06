import sys, os
import cv2
import numpy as np
import json
import trimesh
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, "/home/shuosha/projects/FoundationPose")
from Utils import draw_xyz_axis # type: ignore
from estimater import FoundationPose # type: ignore

import logging
logging.getLogger().setLevel(logging.WARNING)

H, W = 480, 848
def get_mask(initial_scene, background_scene, corners_list): 
    """
    This function should return a mask based on the initial scene and the background scene.
    For now, we will return None as a placeholder.
    """
    # --- 1) Load images ----------------------------------------
    color  = initial_scene        # your current RGB frame
    bg_bgr = background_scene   # your background image (same size)

    H, W = color.shape[:2]

    # --- 2) Background subtraction mask ----------------------
    #    a) absolute difference & grayscale
    diff = cv2.absdiff(color, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #    b) threshold to binary (0/1)
    _, mask_bs = cv2.threshold(gray, 30, 1, cv2.THRESH_BINARY)

    #    c) clean small holes / speckles
    kernel  = np.ones((5,5), np.uint8)
    mask_bs = cv2.morphologyEx(mask_bs, cv2.MORPH_CLOSE, kernel)

    # --- 3) Quadrilateral ROI mask ----------------------------
    corners = corners_list

    quad_mask = np.zeros((H, W), dtype=np.uint8)
    pts       = np.array(corners, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(quad_mask, [pts], color=1)

    # --- 4) Combine masks --------------------------------------
    #    final mask is 1 only where both are 1
    mask_final = (mask_bs & quad_mask).astype(np.uint8)
    return mask_final

class StateEstimator():
    def __init__(self, foundation_pose_dir=None):
        super().__init__()
        assert foundation_pose_dir is not None, "Foundation pose directory must be provided."
        self.foundation_pose_dir = foundation_pose_dir
        # Initialize other necessary variables and models here

        obj_list = os.path.join(foundation_pose_dir, "object_list.txt")
        with open(obj_list, 'r') as f:
            self.object_list = [line.strip() for line in f.readlines()]
            print(f"Loaded object list: {self.object_list}")
        self.num_objects = len(self.object_list)

        self.empty_scene = cv2.imread(os.path.join(foundation_pose_dir, "empty_scene.png"))
        with open(os.path.join(foundation_pose_dir, "extrinsics.json"), 'r') as f:
            front2base = json.load(f)["cam2base"]["front2base"]
        with open(os.path.join(foundation_pose_dir, "intrinsics.json"), 'r') as f:
            intrinsics = json.load(f)["front"]
        self.cam_k = np.array([
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0]
        ])
        self.cam2base = np.array(front2base).reshape(4, 4)
        with open(os.path.join(foundation_pose_dir, "corners.json"), "r") as f:
            self.corners_dict = json.load(f)

        self.estimators = {}

        for i in range(self.num_objects):
            obj_name = self.object_list[i]
            mesh = trimesh.load(os.path.join(foundation_pose_dir, f"{obj_name}.obj"))
            self.estimators[obj_name] = FoundationPose(
                model_pts=mesh.vertices, 
                model_normals=mesh.vertex_normals, 
                mesh=mesh, 
                debug_dir=foundation_pose_dir, 
                debug=0
            )

        self.cur_object_poses_q = np.array([0.0] * (self.num_objects * 16), dtype=np.float32)

    def estimate_object_poses(self, bgr, depth, retrack=False):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = depth / 1000.0 # convert to meters
        for i in range(self.num_objects):
            obj_name = self.object_list[i]
            # todo add restart estimator if lost
            if self.estimators[obj_name].pose_last is None or retrack:
                corners = self.corners_dict[obj_name]
                mask = get_mask(bgr, self.empty_scene, corners)
                # cv2.imwrite(f"masks_{obj_name}.png", mask * 255)
                # cv2.imwrite(f"rgb_{obj_name}.png", self.rgb)
                # cv2.imwrite(f"depth_{obj_name}.png", depth)
                obj2cam = self.estimators[obj_name].register(
                    K=self.cam_k,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=5,
                )
            else:
                obj2cam = self.estimators[obj_name].track_one(
                    K=self.cam_k,
                    rgb=rgb,
                    depth=depth,
                    iteration=2,
                )
            self.cur_object_poses_q[i*16:(i+1)*16] = obj2cam.flatten().tolist()
        
        return self.cur_object_poses_q.reshape(self.num_objects, 4, 4)

    def draw_detected_objects(self, bgr_image, obj2cam):
        bgr = bgr_image.copy()
        vis = draw_xyz_axis(
            color=bgr,
            ob_in_cam=obj2cam,
            K=self.cam_k,
            is_input_rgb=False,
            scale=0.05,
        )

        # Implement object detection and drawing logic here
        return vis
    
    def project_points_on_image(self, image, points_base):
        """
        image: (H,W,3) uint8
        points_base: (N,3) in base frame
        """
        # invert T_cam_base to get base->cam
        T_base_cam = np.linalg.inv(self.cam2base)
        R_bc = T_base_cam[:3, :3]  # rotation from base to cam
        t_bc = T_base_cam[:3, 3]   # translation from base to cam

        # 1) base -> camera
        pts_base = points_base.T  # (3,N)
        pts_cam = R_bc @ pts_base + t_bc[:, None]  # (3,N)

        X = pts_cam[0, :]
        Y = pts_cam[1, :]
        Z = pts_cam[2, :]

        # keep only points in front of the camera
        mask = Z > 0
        X, Y, Z = X[mask], Y[mask], Z[mask]

        # 2) project using intrinsics: u = fx*X/Z + cx, v = fy*Y/Z + cy
        fx, fy = self.cam_k[0, 0], self.cam_k[1, 1]
        cx, cy = self.cam_k[0, 2], self.cam_k[1, 2]

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        # round and convert to int
        u = u.astype(int)
        v = v.astype(int)

        h, w = image.shape[:2]
        # optionally filter to only those inside the image
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid]
        v = v[valid]

        # 3) draw on the image
        img_vis = image.copy()
        for px, py in zip(u, v):
            cv2.circle(img_vis, (px, py), 3, (0, 0, 255), -1)  # red dots

        return img_vis 

    def draw_triangle_from_base_points(
        self,
        image: np.ndarray,
        fingertip_pos: np.ndarray,
        base_pos: np.ndarray,
        final_pos: np.ndarray,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Project three 3D points (in base frame) onto the RGB image and draw:
        - fingertip -> base    : blue
        - fingertip -> final   : green
        - base -> final        : red
        """
        img = image.copy()

        # --- 1) Build base->camera transform from cam->base ---
        # T_cam_base maps camera -> base, so invert to get base -> camera
        T_base_cam = np.linalg.inv(self.cam2base)
        R_bc = T_base_cam[:3, :3]  # rotation base->cam
        t_bc = T_base_cam[:3, 3]   # translation base->cam

        def project_point(p_base: np.ndarray):
            """Project a single 3D point in base frame to pixel coords (u,v)."""
            # ensure shape (3,)
            p_base = np.asarray(p_base).reshape(3)

            # base -> camera
            p_cam = R_bc @ p_base + t_bc
            X, Y, Z = p_cam

            if Z <= 0:
                return None  # behind camera, skip

            fx, fy = self.cam_k[0, 0], self.cam_k[1, 1]
            cx, cy = self.cam_k[0, 2], self.cam_k[1, 2]

            u = int(fx * (X / Z) + cx)
            v = int(fy * (Y / Z) + cy)

            h, w = img.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return None  # outside image
            return (u, v)

        # --- 2) Project the three points ---
        pt_f = project_point(fingertip_pos)
        pt_b = project_point(base_pos)
        pt_g = project_point(final_pos)  # "goal"

        # --- 3) Draw segments if both endpoints are valid ---
        # colors are BGR in OpenCV
        if pt_f is not None and pt_b is not None:
            cv2.line(img, pt_f, pt_b, (255, 0, 0), thickness)   # blue: fingertip->base
        if pt_f is not None and pt_g is not None:
            cv2.line(img, pt_f, pt_g, (0, 255, 0), thickness)   # green: fingertip->final
        if pt_b is not None and pt_g is not None:
            cv2.line(img, pt_b, pt_g, (0, 0, 255), thickness)   # red: base->final

        return img





