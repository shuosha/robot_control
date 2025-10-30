import os
import copy
import time
import numpy as np
import transforms3d
from pathlib import Path
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import warnings
warnings.filterwarnings("always", category=RuntimeWarning)

import sapien.core as sapien

from robot_control.utils.utils import get_root
root: Path = get_root(__file__)

_ENGINE = None

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def gripper_raw_to_qpos(raw: float) -> float:
    raw_clipped = np.clip(raw, 0.0, 800.0)
    return (800.0 - raw_clipped) / 800.0

def trans_mat_to_pos_quat(trans_mat: np.ndarray) -> np.ndarray:
    pos = trans_mat[..., :3, 3]
    quat_xyzw = R.from_matrix(trans_mat[..., :3, :3]).as_quat() # [x, y, z, w] order
    quat_wxyz = np.roll(quat_xyzw, 1)   # shifts everything right by 1
    return pos, quat_wxyz

def pose6_to_matrix(pose6: np.ndarray) -> np.ndarray:
    """
    Converts a (6,) pose (translation + angle-axis rotation) into a 4x4 transformation matrix.

    Args:
        pose6: np.ndarray of shape (6,) – [x, y, z, rx, ry, rz]

    Returns:
        T: np.ndarray of shape (4, 4) – SE(3) transformation matrix
    """
    assert pose6.shape == (6,), "Input must be a (6,) array"

    # Translation
    t = pose6[:3]

    # Rotation from angle-axis
    r = R.from_rotvec(pose6[3:])  # scipy takes angle-axis directly
    R_mat = r.as_matrix()         # 3x3 rotation matrix

    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    return T

# ---------- Quaternion helpers (wxyz) ----------
def _q_normalize(q, eps=1e-12):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def _q_conj(q):
    # q = [w, x, y, z]
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def _q_mul(q1, q2):
    # Hamilton product, wxyz
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def _quat_apply(q, v):
    """Rotate vector v by unit quaternion q (wxyz). Shapes broadcast on last dim."""
    w = q[..., :1]          # (..,1)
    xyz = q[..., 1:]        # (..,3)
    uv  = torch.cross(xyz, v, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)

class KinHelper():
    def __init__(self, robot_name, sapien_env_tuple=None, headless=True):
        # load robot
        if "xarm7" in robot_name:
            urdf_path = str(root / "src/robot_control/assets/xarm7/xarm7.urdf")
            self.eef_name = 'link7'
        else:
            raise RuntimeError('robot name not supported')
        self.robot_name = robot_name

        # load sapien robot
        if sapien_env_tuple is not None:
            engine, scene, loader = sapien_env_tuple
        else:
            global _ENGINE
            if _ENGINE is None:
                _ENGINE = sapien.Engine()            # create once
            engine = _ENGINE
            scene = engine.create_scene()
            loader = scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == self.eef_name:
                self.sapien_eef_idx = link_idx
                break

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        self.pcd_dict = {}
        self.tool_meshes = {}

    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def compute_ik_sapien(self, initial_qpos, cartesian, verbose=False):
        """
        Compute IK using sapien
        initial_qpos: (N, ) numpy array
        cartesian: (6, ) numpy array, x,y,z in meters, r,p,y in radians
        """
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes='sxyz')
        tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf_mat)

        if 'xarm7' in self.robot_name:
            active_qmask = np.array([True, True, True, True, True, True, True])
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx, 
            pose=pose,
            initial_qpos=initial_qpos, 
            active_qmask=active_qmask, 
            )
        if verbose:
            print('ik qpos:', qpos)

        # verify ik
        fk_pose = self.compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        
        if verbose:
            print('target pose for IK:', tf_mat)
            print('fk pose for IK:', fk_pose)
        
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf_mat[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf_mat[:3, :3])
        
        if pose_diff > 0.01 or rot_diff > 0.01:
            print('ik pose diff:', pose_diff)
            print('ik rot diff:', rot_diff)
            warnings.warn('ik pose diff or rot diff too large. Return initial qpos.', RuntimeWarning, stacklevel=2, )
            return initial_qpos
        return qpos[0]


def test_fk():
    robot_name = 'xarm7'
    init_qpos = np.array([0, 0, 0, 0, 0, 0, 0])
    end_qpos = np.array([0, 0, 0, 0, 0, 0, 0])
    
    kin_helper = KinHelper(robot_name=robot_name, headless=False)
    START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0]

    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk = kin_helper.compute_fk_sapien_links(curr_qpos, [kin_helper.sapien_eef_idx])[0]
        fk_euler = transforms3d.euler.mat2euler(fk[:3, :3], axes='sxyz')

        if i == 0:
            init_ik_qpos = np.array(START_ARM_POSE)
        ik_qpos = kin_helper.compute_ik_sapien(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        re_fk_pos_mat = kin_helper.compute_fk_sapien_links(ik_qpos, [kin_helper.sapien_eef_idx])[0]
        re_fk_euler = transforms3d.euler.mat2euler(re_fk_pos_mat[:3, :3], axes='sxyz')
        re_fk_pos = re_fk_pos_mat[:3, 3]
        print('re_fk_pos diff:', np.linalg.norm(re_fk_pos - fk[:3, 3]))
        print('re_fk_euler diff:', np.linalg.norm(np.array(re_fk_euler) - np.array(fk_euler)))
        

        init_ik_qpos = ik_qpos.copy()
        qpos_diff = np.linalg.norm(ik_qpos[:6] - curr_qpos[:6])
        if qpos_diff > 0.01:
            warnings.warn('qpos diff too large', RuntimeWarning, stacklevel=2, )

        time.sleep(0.1)

if __name__ == "__main__":
    test_fk()
