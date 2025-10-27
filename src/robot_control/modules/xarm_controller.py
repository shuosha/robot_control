import threading
import multiprocess as mp
import time
from enum import Enum
import traceback

import numpy as np
import transforms3d
import copy

from xarm.wrapper import XArmAPI

from robot_control.utils.kinematics_utils import KinHelper
from robot_control.utils.udp_util import udpReceiver, udpSender
from robot_control.modules.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from robot_control.modules.common.xarm import *

np.set_printoptions(precision=2, suppress=True)


class Rate:
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


class ControllerState(Enum):
    INIT = 0
    RUNNING = 1
    STOP = 2

class XarmController(mp.Process):
    
    name = "xarm_controller"
    teleop_kin_helper = KinHelper(robot_name="xarm7")
    robot_kin_helper = KinHelper(robot_name="xarm7")

    XY_MIN, XY_MAX = XY_MIN, XY_MAX 
    X_MIN, X_MAX = X_MIN, X_MAX 
    Y_MIN, Y_MAX = Y_MIN, Y_MAX 
    Z_MIN, Z_MAX = Z_MIN, Z_MAX 
    
    MOVE_RATE = MOVE_RATE 
    MOVE_SLEEP = MOVE_SLEEP 
    XYZ_VELOCITY = XYZ_VELOCITY 
    ANGLE_VELOCITY_MAX = ANGLE_VELOCITY_MAX 

    GRIPPER_OPEN_MAX = GRIPPER_OPEN_MAX
    GRIPPER_OPEN_MIN = GRIPPER_OPEN_MIN

    POSITION_UPDATE_INTERVAL = POSITION_UPDATE_INTERVAL 
    COMMAND_CHECK_INTERVAL = COMMAND_CHECK_INTERVAL 


    def log(self, msg):
        if self.verbose:
            self.pprint(msg)
    
    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(
                "[{}][{}] {}".format(
                    time.strftime("\033[92m%Y-%m-%d %H:%M:%S\033[0m", time.localtime(time.time())),
                    stack_tuple[1],
                    " ".join(map(str, args)),
                )
            )
        except:
            print(*args, **kwargs)

    def __init__(
        self,
        start_time,
        init_pose=[256.7, 5.1, 400.1, 178.9, 0.0, 1.4],
        init_servo_angle=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0],
        ip="192.168.1.196", 
        robot_id=0,
        control_mode="position_control",  # "position_control" or "velocity_control"
        robot_name="xarm7", # "xarm7" or "uf850"
        teleop_robot_model_name="xarm7",
        admittance_control=False,
        ema_factor=1.0,
        gripper_enable=False,
        speed=50,  # mm/s
        verbose=False,
    ):
        
        self.robot_id = robot_id
        self.init_pose = init_pose
        self.init_servo_angle = init_servo_angle
        self.speed = speed
        self.control_mode = control_mode
        assert control_mode in ["position_control", "velocity_control", "cartesian_position_planning"], "control_mode must be position_control, velocity_control or cartesian_position_planning"
        self.admittance_control = admittance_control
        if admittance_control:
            assert control_mode != "velocity_control", "admittance control is not compatible with velocity control mode"
        self.ema_factor = ema_factor

        assert robot_name and teleop_robot_model_name in ["xarm7", "uf850"], "robot_name and teleop_robot_model_name must be xarm7 or uf850"
        self.mismatch_kinematics = (robot_name != teleop_robot_model_name)

        # assert mode == "2D", "currently only support 2D mode"
        super().__init__()
        
        self.start_time = start_time
        self._ip = ip
        self.gripper_enable = gripper_enable
        # input(f"Connecting to {self._ip}, Press Enter to continue...")
        self.verbose = verbose
        self.exe_lock = mp.Lock()

        self.state = mp.Value('i', ControllerState.INIT.value)

        # self.command_q = mp.Queue()
        self.cur_trans_q = mp.Array('d', [0.0] * 16)
        self.cur_qpos_q = mp.Array('d', [0.0] * 7)
        self.cur_gripper_q = mp.Array('d', [0.0] * 1)
        self.cur_force_q = mp.Array('d', [0.0] * 6)
        self.cur_time_q = mp.Value('d', 0.0)

        self.command_receiver = None
        # self.state_sender = None

        self.cur_gripper_pos = None
        self.cur_qpos = None

        self.teleop_activated = mp.Value('b', False)  # whether the teleop is activated, used to control the gripper

    # ======= xarm controller queue START =======
    def update_cur_position(self):
        """ update the current position of the arm in a separate thread,
        due to the unprecise of get_position API, use sapien fk to do the position closed loop"""

        if self.robot_id == -1:
            self.state_sender = udpSender(port=XARM_STATE_PORT)
        try:
            while self.state.value in [ControllerState.INIT.value, ControllerState.RUNNING.value]:
                # update_start_time = time.time()

                if self.gripper_enable:
                    cur_gripper_pos = self.get_gripper_state()
                    self.cur_gripper_q[:] = np.array([cur_gripper_pos])
                    self.cur_gripper_pos = cur_gripper_pos

                cur_qpos = np.array(self._arm.get_servo_angle()[1][0:7]) / 180. * np.pi
                
                current_time = time.time()
                self.cur_time_q.value = current_time

                fk_trans_mat = self.robot_kin_helper.compute_fk_sapien_links(cur_qpos, [self.robot_kin_helper.sapien_eef_idx])[0]
                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:3] = fk_trans_mat[:3, 3] * 1000
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
                cur_xyzrpy[3:] = cur_xyzrpy[3:] / np.pi * 180
                
                # self.log(f"d_pos: {self.arm.get_position()[1][0:6]-cur_position}")

                # always update the latest position
                self.cur_trans_q[:] = fk_trans_mat.flatten()
                self.cur_qpos_q[:] = cur_qpos

                code, ft_data = self._arm.get_ft_sensor_data()
                if code == 0:
                    self.cur_force_q[:] = np.array(ft_data)
                else:
                    raise ValueError("get_ft_sensor_data Error", code, ft_data)

                # self.cur_xyzrpy = cur_xyzrpy
                self.cur_qpos = cur_qpos
                # print(f"Get xarm_control data in {time.time() - update_start_time:.6f} seconds")
                # time.sleep(max(0, self.POSITION_UPDATE_INTERVAL - (time.time() - update_start_time)))

                if self.robot_id == -1:
                    state = {
                        # "e2b": fk_trans_mat,
                        "pos": cur_xyzrpy,
                        "qpos": cur_qpos,
                    }
                    if self.gripper_enable:
                        state["gripper"] = cur_gripper_pos
                    self.state_sender.send(state)

        except Exception as e:
            self.pprint(f"update_cur_position error")
            self.state.value = ControllerState.STOP.value
        finally:
            # self.state_sender.close()
            # self.command_q.close()
            # self.cur_trans_q.close()
            # self.cur_qpos_q.close()
            # self.cur_gripper_q.close()
            print("update_cur_position exit!")

    def get_current_joint(self):
        return copy.deepcopy(self.cur_qpos)
    
    def get_current_gripper(self):
        return copy.deepcopy(self.cur_gripper_pos)

    def get_current_joint_deg(self):
        return copy.deepcopy(self.cur_qpos) / np.pi * 180

    def get_current_pose(self):
        raise NotImplementedError
        return self.cur_xyzrpy

    # ======= xarm controller queue END =======


    # ======= xarm SDK API wrapper START =======
    def open_gripper(self, wait=True):
        return self.set_gripper_openness(self.GRIPPER_OPEN_MAX, wait=wait)

    def close_gripper(self, wait=True):
        return self.set_gripper_openness(self.GRIPPER_OPEN_MIN, wait=wait)

    # @DeprecationWarning
    def set_gripper_openness(self, openness, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(openness, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("close_gripper Error")
        return True

    def get_gripper_state(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, state = self._arm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            raise ValueError("get_gripper_position Error")
        return state

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._arm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._arm.release_state_changed_callback(self._state_changed_callback)
    
    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            # import ipdb; ipdb.set_trace()
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._arm.connected,
                    self._arm.state,
                    self._arm.error_code,
                    ret1,
                    ret2,
                )
            )
        return self.is_alive
    # ======= xarm SDK API wrapper END =======

    # ======= xarm control START =======
    def _init_robot(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        if self.control_mode == "position_control":
            mode = 1 # streaming position control == servo mode
        elif self.control_mode == "velocity_control":
            mode = 4 # streaming velocity control
        elif self.control_mode == "cartesian_position_planning":
            mode = 0 # cartesian position planning
        else:
            raise ValueError("Invalid control mode")
        self._arm.set_mode(mode)  # NOTE: 0: position control mode, 1: servo control mode, 4: velocity control mode
        self._arm.set_state(0)
        if self.gripper_enable:
            self._arm.set_gripper_enable(True)
            self._arm.set_gripper_mode(0)
            self._arm.clean_gripper_error()
        # self._arm.set_collision_sensitivity(1)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, "register_count_changed_callback"):
            self._arm.register_count_changed_callback(self._count_changed_callback)

        if self.admittance_control:
            assert self.control_mode != "velocity_control", "admittance control is not compatible with velocity control mode"
            # set tool admittance parameters:
            K_pos = 500       #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
            K_ori = 4           #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad)

            # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful. 
            M = float(0.02)  #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
            J = M * 0.01     #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

            c_axis = [0,0,1,0,0,0] # set z axis as compliant axis
            ref_frame = 0         # 0 : base , 1 : tool

            self._arm.set_ft_sensor_admittance_parameters(coord=ref_frame, c_axis=c_axis, M=[M, M, M, J, J, J], K=[K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], B=[0]*6) # B(damping) is reserved, give zeros

            # enable ft sensor communication
            self._arm.set_ft_sensor_enable(1)
            # will overwrite previous sensor zero and payload configuration
            self._arm.set_ft_sensor_zero() # remove this if zero_offset and payload already identified & compensated!
            time.sleep(0.2) # wait for writing zero operation to take effect, do not remove #TODO: test longer time

            # move robot in admittance control application
            self._arm.set_ft_sensor_mode(1)
            # will start after set_state(0)
            self._arm.set_state(0)

            print("ft sensor cfg: ", self._arm.get_ft_sensor_config())
            # print("ft collision reb distance: ", self._arm.get_ft_collision_reb_distance()) #TODO: DEBUG
            print("ft collision rebound: ", self._arm.get_ft_collision_rebound())
            # print("ft collision threshold: ", self._arm.get_ft_collision_threshold())
        
        self.state.value = ControllerState.RUNNING.value

    def reset(self):
        # return self._reset()  # position control
        # return self._reset_pose()  # NOTE: servo control, use sapien ik to move
        return
    
    def _reset(self, wait=True):
        self.move_to_pose(self.init_pose, wait=wait)
        self._arm.set_servo_angle(angle=self.init_servo_angle, isradian=False, wait=wait)
        if self.gripper_enable:
            self.open_gripper(wait=wait)

    def _reset_pose(self):
        # init pose
        if not self.exe_lock.acquire(block=True, timeout=1):
            self.log("xarm reset failed! exe_lock not acquired!")
            return
        
        self.move(self.init_pose, steps=500, clean=True)
        self.exe_lock.release()

    def check_valid_move(self, next_position, steps):
        # absolute position
        if len(next_position) == 6:
            x, y, z, roll, pitch, yaw = next_position
        elif self.gripper_enable and len(next_position) == 7:
            x, y, z, roll, pitch, yaw, gripper_pos = next_position
            if gripper_pos < self.GRIPPER_OPEN_MIN or gripper_pos > self.GRIPPER_OPEN_MAX:
                self.log(f"invalid move command {next_position}! gripper out of range!")
                return False
        
        if x ** 2 + y ** 2 > self.XY_MAX ** 2 or x ** 2 + y ** 2 < self.XY_MIN ** 2\
            or x < self.X_MIN or x > self.X_MAX or y < self.Y_MIN or y > self.Y_MAX:
            self.log(f"invalid move command {next_position}! x,y out of range!")
            return False
        elif z > self.Z_MAX or z < self.Z_MIN:
                self.log(f"invalid move command {next_position}! z out of range!")
                return False

        return True

    def velocity_control(self, next_joints, current_joints, ema_factor=1.0, ignore_error=False):
        """
        streaming velocity targets
        """
        # NOTE: velocity control don't use ema
        # next_joints = ema_factor * next_joints + (1 - ema_factor) * current_joints

        # NOTE: delta for velocity control
        next_joints[0:7] = next_joints[0:7] - current_joints[0:7]
        
        # denormalize gripper position
        if self.gripper_enable:
            gripper_pos = next_joints[-1]
            denormalized_gripper_pos = gripper_pos * (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX) + GRIPPER_OPEN_MAX
            next_joints[-1] = denormalized_gripper_pos

        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        if self.gripper_enable and len(next_joints) == 7: #TODO: hardcoded right now
            if not np.isclose(self.cur_gripper_pos, next_joints[-1]):
                self.set_gripper_openness(next_joints[-1], wait=False)
            next_joints = next_joints[:-1]

        v = next_joints / self.COMMAND_CHECK_INTERVAL * 0.15
        v = v.tolist()
        # print("set velocity:", np.round(v,4), "shape:", np.array(v).shape)
        code = self._arm.vc_set_joint_velocity(v, is_radian=True, is_sync=False, duration=0)

        if not self._check_code(code, "vc_set_joint_velocity"):
            raise ValueError("velocity control error")
        if ignore_error:
            self._arm.clean_error()
            self._arm.clean_warn()

    def position_control(self, next_state, current_state, ema_factor=1.0, ignore_error=False):
        """
        streaming position/servo targets
        """
        # ema
        next_joints = ema_factor * next_state[:7] + (1 - ema_factor) * current_state[:7]
        # print("next joints (rad):", np.round(next_joints,4), "shape:", np.array(next_joints).shape)
        if self.gripper_enable:
            gripper_pos = next_state[-1]
        
        # denormalize gripper position
        if self.gripper_enable:
            gripper_pos = next_state[-1]
            denormalized_gripper_pos = gripper_pos * (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX) + GRIPPER_OPEN_MAX
            # print("denormalized_gripper_pos:", denormalized_gripper_pos)

        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        if self.gripper_enable and len(next_joints) == 7: #TODO: hardcoded right now
            if not np.isclose(self.cur_gripper_pos, denormalized_gripper_pos):
                self.set_gripper_openness(denormalized_gripper_pos, wait=False)

        next_joints = next_joints.tolist()
        code = self._arm.set_servo_angle_j(angles=next_joints, is_radian=True, speed=1.0, acc=None, wait=False)

        if not self._check_code(code, "set_servo_angle_j"):
            raise ValueError("position control error")
        if ignore_error:
            self._arm.clean_error()
            self._arm.clean_warn()

    # def move_joints(self, next_joints, wait=False, ignore_error=False):
    #     assert wait == False, "wait is not supported in move_joints"
    #     if not self.is_alive:
    #         raise ValueError("Robot is not alive!")
    #     if self.gripper_enable and len(next_joints) == 7: #TODO: hardcoded right now
    #         if not np.isclose(self.cur_gripper_pos, next_joints[-1]):
    #             self.set_gripper_openness(next_joints[-1], wait=wait)
    #         next_joints = next_joints[:-1]
        
    #     # # velocity control (next_joints needs to be delta)
    #     # v = next_joints / self.COMMAND_CHECK_INTERVAL * 0.15
    #     # v = v.tolist()
    #     # print("set velocity:", np.round(v,4), "shape:", np.array(v).shape)
    #     # code = self._arm.vc_set_joint_velocity(v, is_radian=True, is_sync=False, duration=0)
        
    #     # position control
    #     next_joints = next_joints.tolist()
    #     code = self._arm.set_servo_angle_j(angles=next_joints, is_radian=True, speed=1.0, acc=None, wait=wait)
        
    #     # if not self._check_code(code, "vc_set_joint_velocity"):
    #     #     raise ValueError("move_joints Error")
    #     if not self._check_code(code, "set_servo_angle_j"):
    #         raise ValueError("move_joints Error")
    #     if ignore_error:
    #         self._arm.clean_error()
    #         self._arm.clean_warn()

    def move_to_pose(self, pose, wait=False, ignore_error=False):
        return self.move(pose)

    def move(self, next_position, steps=10, clean=True):
        self._move_ik(next_position, steps, clean)  # NOTE: use sapien ik to move
        # self._move(next_position, steps, clean)

    def _move(self, pose, steps, clean):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_position(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], speed=self.speed, wait=True
        )
        if not self._check_code(code, "set_position"):
            raise ValueError("move_to_pose Error")
        if clean:
            self._arm.clean_error()
            self._arm.clean_warn()

    def _move_ik(self, next_position, steps, clean):
        """next_position :  x,y,z in mm   and   r,p,y in degree  [and gripper]"""
        assert next_position is not None, "next_position is not set!"
        # next_position = np.array(next_position)
        if not self.check_valid_move(next_position, steps):
            print(f"invalid move command {next_position}!")
            return
        
        self.log(f'move start: {next_position}')

        if self.gripper_enable and len(next_position) == 7:
            if not np.isclose(self.cur_gripper_pos, next_position[-1]):
                self.set_gripper_openness(next_position[-1])
            next_position = next_position[:-1]

        initial_qpos = np.array(self._arm.get_servo_angle()[1][0:6]) / 180. * np.pi
        next_position_m_rad = np.zeros_like(np.array(next_position))
        next_position_m_rad[0:3] = np.array(next_position)[0:3] / 1000.
        next_position_m_rad[3:] = np.array(next_position)[3:] / 180. * np.pi
        next_servo_angle = self.kin_helper_uf850.compute_ik_sapien(initial_qpos, next_position_m_rad)

        # fix the eef joint to [-pi,pi]
        next_servo_angle[-1] = (next_servo_angle[-1] + np.pi) % (2 * np.pi) - np.pi

        # NOTE: In the servo mode: state(1), the actual speed is contorlled by the
        #       rate of command sending and the the distance between the current and target position 
        # The steps of each move is decided by the distance of moving to keep speed constant
        init_position = self.kin_helper_uf850.compute_fk_sapien_links(initial_qpos, [self.kin_helper_uf850.sapien_eef_idx])[0][:3,3] * 1000
        dis_diff = np.array(next_position[:3]) - np.array(init_position)
        distance = np.linalg.norm(dis_diff) # in millimeter
        min_steps = int(distance / self.XYZ_VELOCITY)
        self.log(f"distance: {distance}, min_steps: {min_steps}")
        steps = max(min_steps, steps)

        # TODO: add max angular velocity control
        angle_diff = np.array(next_servo_angle) - np.array(initial_qpos)
        angle_distance = np.max(abs(angle_diff)) *180 / np.pi
        self.log(f"angle_distance: {angle_distance}")
        self.log(f"last steps: {steps}, angle ref steps: {int(angle_distance / self.ANGLE_VELOCITY_MAX)}")
        steps = max(steps, int(angle_distance / self.ANGLE_VELOCITY_MAX))

        tic = time.time()
        for i in range(steps): 
            angle = initial_qpos + (next_servo_angle - initial_qpos) * (i + 1) / steps
            if not self.is_alive:
                raise ValueError("Robot is not alive!")
            code = self._arm.set_servo_angle_j(angles=angle, is_radian=True, speed=1)
            if not self._check_code(code, "set_position"):
                raise ValueError("move Error")
            time.sleep(self.MOVE_SLEEP)

        self.log(f"move end: volecity: {distance/(time.time()-tic):.2f} mm/s")

        if clean:
            self._arm.clean_error()
            self._arm.clean_warn()

    # ======= xarm control END =======

    # ======= main thread loop =======
    def run(self):
        # the arm initialization must be invoked in the same process
        print("Connecting to xarm at", self._ip)
        self._arm = XArmAPI(self._ip)
        print("Connected to xarm at", self._ip)
        if self.verbose:
            self.log(f"Connected to xarm at {self._ip}")
        self._init_robot()

        # NOTE: the arm can only be interacted with the process created the API class, 
        self.update_pos_t = threading.Thread(target=self.update_cur_position, name="update_cur_position")
        self.update_pos_t.start()

        port = XARM_CONTROL_PORT_L if self.robot_id <= 0 else XARM_CONTROL_PORT_R  # -1: keyboard, 0: gello left, 1: gello right
        self.command_receiver = udpReceiver(ports={'xarm_control': port})
        self.command_receiver.start()

        print(f"{'='*20} xarm start! state: {ControllerState(self.state.value)}")
        self.reset()
        print(f"{'='*20} xarm reset! state: {ControllerState(self.state.value)}")
        
        command = None
        self.teleop_activated.value = False #if self.control_mode == 'joints' else True
        rate = Rate(
            duration=COMMAND_CHECK_INTERVAL,
        )

        while self.state.value == ControllerState.RUNNING.value:
            try:
                start_time = time.time()
                commands = self.command_receiver.get("xarm_control", pop=True)
                command_timestamp = time.time()
                if commands is None or len(commands[0]) == 0:
                    if self.control_mode == 'velocity_control':
                        self._arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0, 0], is_radian=True, is_sync=False, duration=0)
                    elif self.control_mode == 'position_control':
                        self._arm.set_servo_angle_j(angles=self._arm.get_servo_angle(is_radian=True)[1][:7], is_radian=True, speed=1.0, wait=False)
                    elif self.control_mode == 'cartesian_position_planning':
                        self._arm.set_position(x=500, y=0,z=300,roll=180,pitch=0,yaw=-90, speed=100, wait=True) # NOTE: hardcoded
                    rate.sleep()
                    # time.sleep(max(0, self.COMMAND_CHECK_INTERVAL - (time.time() - start_time)))
                    continue

                print_commands = False
                if len(commands[0]) > 0 and print_commands:
                    if self.robot_id == 1:
                        print('\t' * 12, end='')
                    print(f'activated: {self.teleop_activated.value}, commands: {[np.round(c, 4) for c in commands[0]]}')
                # continue  # enable for debug

                with self.exe_lock:
                    self.log("xarm controller get lock")
                    self.log(f"new commands at {time.time() - self.start_time:.2f}")
                    
                    # NOTE: each robot command is a element in commands
                    # NOTE: command always teleop device in joint space
                    for command in commands:
                        self.log(f"get command: {command}")

                        if isinstance(command, str):
                            if command == "quit":
                                break

                            # elif command == "reset":  # TODO disable reset
                            #     command = self.init_pose
                        
                        # convert to joint space target
                        if self.control_mode == 'cartesian_position_planning':
                            assert self.mismatch_kinematics == False, "currently cartesian position planning mode does not support mismatch kinematics"
                            command_state = np.array(command) # joint space + gripper
                            fk = self.robot_kin_helper.compute_fk_sapien_links(command_state[:7], [self.robot_kin_helper.sapien_eef_idx])[0] # compute fk with xarm7

                            fk_euler = transforms3d.euler.mat2euler(fk[:3, :3], axes='sxyz')
                            command = np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32)

                            if self.gripper_enable:
                                command = np.concatenate([command, command_state[-1:]]) # cartesian space + gripper

                            # low level control
                            self.cartesian_position_control(command)

                        elif self.control_mode == 'velocity_control' or self.control_mode == 'position_control':
                            command_state = np.array(command)
                            assert len(command_state) == 8, "received gello-xarm qpos must be 8-dim"

                            # obtain current robot state
                            current_joints = np.array(self._arm.get_servo_angle()[1][0:7]) / 180. * np.pi
                            if self.gripper_enable:
                                current_gripper = self.get_gripper_state()
                                current_gripper = (current_gripper - GRIPPER_OPEN_MAX) / (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX)
                                current_state = np.concatenate([current_joints, np.array([current_gripper])])
                            else:
                                current_state = current_joints

                            if self.mismatch_kinematics:
                                # robot_cart = self.robot_kin_helper.compute_fk_sapien_links(current_joints, [self.robot_kin_helper.sapien_eef_idx])[0] # compute fk with xarm7
                                teleop_cart = self.teleop_kin_helper.compute_fk_sapien_links(command_state[:7], [self.teleop_kin_helper.sapien_eef_idx])[0] # compute fk with xarm7
                                teleop_cart_euler = transforms3d.euler.mat2euler(teleop_cart[:3, :3], axes='sxyz')
                                cart_comm = np.array(list(teleop_cart[:3, 3]) + list(teleop_cart_euler)).astype(np.float32)
                                # print('fk pos:', teleop_cart[:3, 3], 'fk euler:', np.array(teleop_cart_euler) / np.pi * 180, "gripper", command_state[-1])
                                command = self.robot_kin_helper.compute_ik_sapien(current_joints, cart_comm) # compute ik with ur850
                                if self.gripper_enable:
                                    command = np.concatenate([command, command_state[-1:]])

                            # check teleop activated
                            delta = command_state - current_state # joint space + gripper
                            joint_delta_norm = np.linalg.norm(delta[0:7])
                            max_joint_delta = np.abs(delta[0:7]).max()
                            # print('teleop activated:', self.teleop_activated.value, 'command latency:', time.time() - command_timestamp, 'command_state:', command_state, 'current_state:', current_state)

                            max_activate_delta = 0.5
                            max_delta_norm = 0.10
                            if not self.teleop_activated.value:
                                if max_joint_delta < max_activate_delta:
                                    self.teleop_activated.value = True
                                next_state = current_state
                            else:
                                if joint_delta_norm > max_delta_norm:
                                    delta[0:7] = delta[0:7] / joint_delta_norm * max_delta_norm # upper bounds delta at: max_delta_norm
                                next_state = current_state + delta

                            # print('next_state:', next_state)
                            # communicate to low-level control
                            if self.control_mode == 'position_control':
                                self.position_control(next_state, current_state, self.ema_factor)
                            elif self.control_mode == 'velocity_control':
                                self.velocity_control(next_state, current_state, self.ema_factor)

                    if command == "quit":
                        break
                
                rate.sleep()
                # time.sleep(max(0, self.COMMAND_CHECK_INTERVAL - (time.time() - start_time)))
            
            except BaseException as e:
                self.log(f"Error in xarm controller: {e.with_traceback()}")
                break

        self.stop()
        self.command_receiver.stop()
        print("xarm controller stopped")

    # ======= process control =======
    # Only the process created the API class can start and control the robot, init in the `run` function 
    
    def start(self) -> None:
        return super().start()
    
    def stop(self):
        self.state.value = ControllerState.STOP.value
        self._arm.set_ft_sensor_mode(0)
        self._arm.set_ft_sensor_enable(0)
        self._arm.disconnect()
        print(f"{'='*20} xarm exit! state: {ControllerState(self.state.value)}")
        # self.log(f"{'='*20} xarm exit! state: {ControllerState(self.state.value)}")

    @property
    def is_alive(self):
        """check whether the robot and the controller is alive
        To check only the controller condition, use self.state.value == ControllerState.RUNNING.value"""
        if self.is_controller_alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    @property
    def is_controller_alive(self):
        return self.state.value == ControllerState.RUNNING.value

    
if __name__ == "__main__":

    start_time = time.time()
    controller = XarmController(
        start_time=start_time,
        gripper_enable=False,
        z_plane_height=434,
        verbose=True,)
    controller.start()
    controller.join()
