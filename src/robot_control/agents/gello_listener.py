import threading
import multiprocess as mp
from multiprocess.managers import SharedMemoryManager
import time
import numpy as np
import copy
from pynput import keyboard
from pathlib import Path
from typing import Tuple, List
import transforms3d

from robot_control.utils.udp_util import udpReceiver, udpSender
from robot_control.utils.kinematics_utils import KinHelper

from robot_control.modules.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from robot_control.modules.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX
from robot_control.camera.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from third_party.gello.agents.agent import BimanualAgent, DummyAgent
from third_party.gello.agents.gello_agent import GelloAgent
from third_party.gello.env import RobotEnv
from third_party.gello.zmq_core.robot_node import ZMQClientRobot
from third_party.gello.agents.gello_agent import DynamixelRobotConfig
from third_party.gello.dynamixel.driver import DynamixelDriver

np.set_printoptions(precision=2, suppress=True)


class GelloListener(mp.Process):
    def __init__(
        self, 
        # shm_manager: SharedMemoryManager, 
        bimanual: bool = False,
        gello_port: str = '/dev/ttyUSB0',
        bimanual_gello_port: List[str] = ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        baudrate: int = 57600,
    ):
        super().__init__()
        
        self.bimanual = bimanual
        self.bimanual_gello_port = bimanual_gello_port

        self.num_joints = 7
        self.gello_port = gello_port
        self.baudrate = baudrate
        self.do_offset_calibration = False  # whether to recalibrate the offset
        self.verbose = True
        self.initialize_done = mp.Value('b', True)

        if bimanual:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['left_timestamp'] = 0.0
            # examples['right_timestamp'] = 0.0
        else:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['timestamp'] = 0.0

        # ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=examples,
        #     get_max_k=30,
        #     get_time_budget=0.2,
        #     put_desired_frequency=100,
        # )
        self.command = mp.Array('d', examples['command'])
        # self.ring_buffer = ring_buffer
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    # def start(self, wait=True):
    #     super().start()
    #     if wait:
    #         self.start_wait()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.end_wait()

    # def start_wait(self):
    #     self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self): # , k=None, out=None):
        return copy.deepcopy(np.array(self.command[:]))
        # if k is None:
        #     return self.ring_buffer.get(out=out)
        # else:
        #     return self.ring_buffer.get_last_k(k, out=out)

    def init_gello(self):
        if self.bimanual:
            if self.do_offset_calibration:
                assert len(self.bimanual_gello_port) == 2, "Please provide two ports for bimanual calibration"
                left_joint_offsets, left_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[0])
                right_joint_offsets, right_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[1])
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=left_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=left_gripper_config,
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=right_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=right_gripper_config,
                )
            else:
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        4 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 288, 246),
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        3 * np.pi / 2,
                        1 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        0 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 114, 72),
                )
            left_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            right_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            left_agent = GelloAgent(port=self.bimanual_gello_port[0], dynamixel_config=dynamixel_config_left, start_joints=left_start_joints)
            right_agent = GelloAgent(port=self.bimanual_gello_port[1], dynamixel_config=dynamixel_config_right, start_joints=right_start_joints)
            agent = BimanualAgent(left_agent, right_agent)
            self.agent = agent

        else:
            if self.do_offset_calibration:
                joint_offsets, gripper_config = self.calibrate_offset(port=self.gello_port)
            
            else:
                if self.gello_port == '/dev/ttyUSB1':
                    joint_offsets = (
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        4 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    )
                    gripper_config = (8, 288, 246)
                else:
                    assert self.gello_port == '/dev/ttyUSB0'
                    joint_offsets = (
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        0 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    )
                    gripper_config = (8, 290, 248)

            dynamixel_config = DynamixelRobotConfig(
                joint_ids=(1, 2, 3, 4, 5, 6, 7),
                joint_offsets=joint_offsets,
                joint_signs=(1, 1, 1, 1, 1, 1, 1),
                gripper_config=gripper_config,
            )
            gello_port = self.gello_port
            start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            try:
                agent = GelloAgent(port=gello_port, dynamixel_config=dynamixel_config, start_joints=start_joints)
            except:
                print(f"Failed to connect to Gello on port {gello_port}")
                try:
                    agent = GelloAgent(port=self.bimanual_gello_port[1], dynamixel_config=dynamixel_config, start_joints=start_joints)
                except:
                    print(f"Failed to connect to Gello on port {gello_port} and {self.bimanual_gello_port[1]}")
                    raise
            self.agent = agent

        self.ready_event.set()
    
    def calibrate_offset(self, port, verbose=False):
        # MENAGERIE_ROOT = Path(__file__).parent / "third_party" / "mujoco_menagerie"
        
        start_joints = tuple(np.deg2rad([0, -45, 0, 30, 0, 75, 0]))  # The joint angles that the GELLO is placed in at (in radians)
        joint_signs = (1, 1, 1, 1, 1, 1, 1)  # The joint angles that the GELLO is placed in at (in radians)

        joint_ids = list(range(1, self.num_joints + 2))
        driver = DynamixelDriver(joint_ids, port=port, baudrate=self.baudrate)

        # assume that the joint state shouold be start_joints
        # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
        # this is done by brute force, we seach in a range of +/- 8pi

        def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
            joint_sign_i = joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = start_joints[index]
            return np.abs(joint_i - start_i)

        for _ in range(10):
            driver.get_joints()  # warmup

        for _ in range(1):
            best_offsets = []
            curr_joints = driver.get_joints()
            for i in range(self.num_joints):
                best_offset = 0
                best_error = 1e6
                for offset in np.linspace(
                    -8 * np.pi, 8 * np.pi, 8 * 4 + 1
                ):  # intervals of pi/2
                    error = get_error(offset, i, curr_joints)
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                best_offsets.append(best_offset)

        gripper_open = np.rad2deg(driver.get_joints()[-1]) - 0.2
        gripper_close = np.rad2deg(driver.get_joints()[-1]) - 42
        if self.verbose:
            print()
            print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
                + " ]",
            )
            print(
                "gripper open (degrees)       ",
                gripper_open,
            )
            print(
                "gripper close (degrees)      ",
                gripper_close,
            )

        joint_offsets = tuple(best_offsets)
        gripper_config = (8, gripper_open, gripper_close)
        return joint_offsets, gripper_config

    def run(self):
        self.init_gello()

        
        while self.alive:
            try:
                curr_time = time.time()
                action = self.agent.get_action()
                self.command[:] = action
                # print("gello update freq: %.2f Hz"%(1.0 / (time.time() - curr_time)))
            except:
                print(f"Error in GelloListener")
                break

        self.stop()
        print("GelloListener exit!")
        
    @property
    def alive(self):
        return not self.stop_event.is_set() and self.ready_event.is_set()

class GelloTeleop(mp.Process): # TODO: make this purely for teleop 

    name="teleop_gello"
    kin_helper = KinHelper(robot_name='xarm7')

    def __init__(
        self, 
        gripper_enable=True,
        bimanual=False,
        teleop_mode=True,
        pusht_mode=False
    ) -> None:
        super().__init__()
        self.gripper_enable = gripper_enable
        if pusht_mode:
            assert not gripper_enable, "Gripper must be disabled for PushT teleop"
        else:
            assert self.gripper_enable, "Gripper must be enabled for Gello teleop"

        self.bimanual = bimanual

        self.key_states = {
            "p": False,
            ",": False,
            ".": False,
            "/": False,
            "r": False,
            "s": False,
        }

        # additional states
        self.init = True
        self.pause = False
        self.reset = mp.Value('b', False)
        self.record_start = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.record_failed = mp.Value('b', False)
        self.keyboard_listener = None
        self.gello_listener = None

        # states for moving the arm
        # self.cur_joints = None
        self.pusht_mode = pusht_mode
        if self.pusht_mode:
            self.command = mp.Array('d', [0.0] * 7)
            self.cur_comm_qpos_q = mp.Array('d', [0.0] * 7)
            self.cur_comm_xy_q = mp.Array('d', [0.0] * 2)

        else:
            self.command = mp.Array('d', [0.0] * 8)  # 7 joints + gripper
            self.cur_comm_qpos_q = mp.Array('d', [0.0] * 7)
            self.cur_comm_trans_q = mp.Array('d', [0.0] * 16)
            self.cur_comm_gripper_q = mp.Array('d', [0.0] * 1)
        
        self.cur_time_q = mp.Value('d', 0.0)

        self._alive = mp.Value('b', True)
        self.controller_quit = True

        # self.state_receiver = None # udpReceiver(ports={'xarm_state':XARM_STATE_PORT})
        self.command_sender = None # udpSender(port=XARM_CONTROL_PORT)
        self.command_sender_left = None
        self.command_sender_right = None

        # self.shm_manager = SharedMemoryManager()
        # self.shm_manager.start()
        # self.gello_activated = False

        self.teleop_mode = teleop_mode

    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            pass

    def on_release(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.esc:
                return False

    def get_command(self):
        # if self.cur_joints is None:
        #     return
        
        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"keyboard teleop pause: {self.pause}")
            time.sleep(0.5)

        if self.key_states[","]:
            self.record_start.value = True
            self.record_failed.value = False
            self.log(f"Record start")
            time.sleep(0.5)
        
        if self.key_states["."]:
            self.record_stop.value = True
            self.log(f"Record stop and success")
            time.sleep(0.5)
        
        if self.key_states["/"]:
            self.record_stop.value = True
            self.record_failed.value = True
            self.log(f"Record stop and failed")
            time.sleep(0.5)

        if self.pause:
            # self.command = []
            return
        else:
            command_joints = self.gello_listener.get()

            if self.pusht_mode:
                fk_trans_mat = self.kin_helper.compute_fk_sapien_links(command_joints[:7], [self.kin_helper.sapien_eef_idx])[0]
                # print("matrix ", fk_trans_mat)
                fk_trans_mat[:3,:3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:2] = fk_trans_mat[:2, 3]
                cur_xyzrpy[2] = 0.22
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])

                self.cur_comm_xy_q[:] = cur_xyzrpy[:2]

                # print(f"cur_xyzrpy: {cur_xyzrpy}")
                next_joints = self.kin_helper.compute_ik_sapien(command_joints[:7], cur_xyzrpy, verbose=False).tolist()
                # print(f"next joints: {next_joints}")

            else:
                next_joints = command_joints.tolist()
                
            self.command[:] = next_joints
            return

    def run(self) -> None:
        if self.teleop_mode:
            self.gello_listener = GelloListener(
                # shm_manager=self.shm_manager,
                bimanual=self.bimanual,
                gello_port='/dev/ttyUSB0',
                bimanual_gello_port=['/dev/ttyUSB0', '/dev/ttyUSB1'],
            )
            self.gello_listener.start()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()

        if self.bimanual:
            self.command_sender_left = udpSender(port=XARM_CONTROL_PORT_L)
            self.command_sender_right = udpSender(port=XARM_CONTROL_PORT_R)
        else:
            self.command_sender = udpSender(port=XARM_CONTROL_PORT)

        time.sleep(1)
        print("Gello teleop start!")

        while self.alive:
            try:
                # command_start_time = time.time()
                # TODO: improve logic here!!!
                current_time = time.time()
                self.cur_time_q.value = current_time
                if self.teleop_mode:
                    self.get_command()
                    if self.pause:
                        command = []
                    else:
                        command = list(self.command)
                else:
                    if self.key_states["r"]:
                        self.reset.value = True
                        self.log(f"keyboard teleop reset: {self.reset.value}, resetting robot to default pose")
                        self.record_stop.value = True
                    if self.key_states["s"]:
                        self.reset.value = False
                        self.log(f"keyboard teleop reset: {self.reset.value}")
                        self.record_start.value = True

                    command = list(self.command)

                qpos = np.array(self.command[:7])

                self.cur_comm_qpos_q[:] = qpos

                if not self.pusht_mode:
                    gripper_qpos = np.array(self.command[7:])

                    self.cur_comm_gripper_q[:] = gripper_qpos

                    fk_trans_mat = self.kin_helper.compute_fk_sapien_links(self.command[:7], [self.kin_helper.sapien_eef_idx])[0]

                    self.cur_comm_trans_q[:] = fk_trans_mat.flatten()

                if self.bimanual:
                    self.command_sender_left.send([self.command[0][0:8]])
                    self.command_sender_right.send([self.command[0][8:16]])
                else:
                    self.command_sender.send([command])
            except Exception as e:
                print(f"Error in GelloTeleop", e.with_traceback())
                break
        
        self.stop()
        if self.bimanual:
            self.command_sender_left.close()
            self.command_sender_right.close()
        else:
            self.command_sender.close()
        self.gello_listener.stop()
        self.keyboard_listener.stop()
        # self.update_joints_t.join()
        print(f"{'='*20} keyboard + gello teleop exit!")
    
    @property
    def alive(self):
        alive = self._alive.value 
        # & \
        #         (self.gello_listener is None or self.gello_listener.is_alive()) & \
        #         (self.keyboard_listener is None or self.keyboard_listener.is_alive())
        self._alive.value = alive
        return alive 

    def stop(self, stop_controller=False):
        if stop_controller:
            self.log("teleop stop controller")
            if self.command_sender is not None:
                self.command_sender.send(["quit"])
            if self.command_sender_left is not None:
                self.command_sender_left.send(["quit"])
            if self.command_sender_right is not None:
                self.command_sender_right.send(["quit"])
            time.sleep(1)
        self._alive.value = False
        self.log("teleop stop")


if __name__ == "__main__":
    gello_teleop = GelloTeleop()
    gello_teleop.start()
    gello_teleop.join()

