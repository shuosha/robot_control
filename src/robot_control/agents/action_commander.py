from typing import Callable, Tuple, Sequence, List, Literal, Optional
import threading
import multiprocess as mp
from multiprocess.managers import SharedMemoryManager
import time
import numpy as np
import copy
from pynput import keyboard
from pathlib import Path
import transforms3d

from robot_control.utils.udp_util import udpReceiver, udpSender
from robot_control.utils.kinematics_utils import KinHelper

from robot_control.modules.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from robot_control.modules.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX
from robot_control.camera.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robot_control.agents.gello_listener import GelloListener

from third_party.gello.agents.agent import BimanualAgent, DummyAgent
from third_party.gello.agents.gello_agent import GelloAgent
from third_party.gello.env import RobotEnv
from third_party.gello.zmq_core.robot_node import ZMQClientRobot
from third_party.gello.agents.gello_agent import DynamixelRobotConfig
from third_party.gello.dynamixel.driver import DynamixelDriver

np.set_printoptions(precision=2, suppress=True)

class ActionAgent(mp.Process):

    name="action_agent"
    kin_helper = KinHelper(robot_name='xarm7')

    def __init__(
        self, 
        bimanual=False,
        joint_space_dim=8,
        action_receiver: Literal["gello", "keyboard", "policy", "replay"]="gello",
    ) -> None:

        """
        class in charge of receiving joint-state actions from teleop sources (gello/keyboard) or policy, and communicating to the robot
        keyboard controls:
            teleop_mode:
                p: pause/unpause teleop
                ,: start recording
                .: stop recording (success)
                /: stop recording (failed)
            non-teleop_mode:
                r: reset robot to default pose
                s: start from default pose
        """

        super().__init__()
        self.bimanual = bimanual
        self.joint_space_dim = joint_space_dim
        if self.bimanual:
            assert self.joint_space_dim == 16, "bimanual joint state dim should be 16 (8 for each arm)"
            self.log("Using bimanual joint mapping mode for teleop")
        else:
            assert self.joint_space_dim == 8 or self.joint_space_dim == 2, "single arm joint state dim dim should be 8 or 2"
            if self.joint_space_dim == 2:
                self.log("Using PushT mode for single arm teleop")
            else:
                self.log("Using joint mapping mode for single arm teleop")

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

        # listeners
        self.action_receiver = action_receiver
        self.keyboard_listener = None
        self.gello_listener = None
        self.policy_listener = None

        # states for moving the arm
        if self.bimanual:
            self.command = mp.Array('d', [0.0] * 16)  # 8d for each arm
            self.cur_qpos_comm = mp.Array('d', [0.0] * 16) 
            self.cur_eef_trans = mp.Array('d', [0.0] * 32)  # 4x4 matrix
        else:
            self.command = mp.Array('d', [0.0] * 8)   # NOTE: always 8d for this gello setup
            self.cur_qpos_comm = mp.Array('d', [0.0] * 8) 
            self.cur_eef_trans = mp.Array('d', [0.0] * 16)  # 4x4 matrix
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

    def log(self, msg):
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
        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"teleop pause status: {self.pause}")
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

        command_joints = self.gello_listener.get() # always 8d or 16d
        assert command_joints.shape[0] in [8, 16], f"gello command shape should be (8,) or (16,), got {command_joints.shape}"

        if self.pause:
            return
        else:
            if self.joint_space_dim == 2:
                # compute eef pose of gello command
                fk = self.kin_helper.compute_fk_sapien_links(command_joints[:7], [self.kin_helper.sapien_eef_idx])[0]
                
                # hardcode rotation and height
                fk[:3,:3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
                fk[3, 3] = 0.22

                # log eef pose
                self.cur_eef_trans = fk.flatten()

                # compute ik to get next joints
                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:3] = fk[:3, 3]
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk[:3, :3])
                next_joints = self.kin_helper.compute_ik_sapien(command_joints[:7], cur_xyzrpy, verbose=False).tolist()

            else:
                # direct joint mapping
                next_joints = command_joints.tolist()
                
            self.command[:] = next_joints
            return

    def run(self) -> None:
        if self.action_receiver == "gello":
            gello_port = '/dev/ttyUSB0'
            baudrate = 57600
            self.gello_listener = GelloListener(
                bimanual=self.bimanual,
                gello_port=gello_port,
                baudrate=baudrate,
                bimanual_gello_port=['/dev/ttyUSB0', '/dev/ttyUSB1'],
            )
            self.log(f"initializing dynamixel gello listener with port: {gello_port} and baudrate: {baudrate}")
            self.gello_listener.start()
        elif self.action_receiver == "policy" or self.action_receiver == "replay":
            # currently directly overwriting command in main process
            pass
        elif self.action_receiver == "keyboard":
            raise NotImplementedError("keyboard only action receiver is not implemented yet")
        else:
            raise ValueError(f"Unknown action receiver type: {self.action_receiver}")

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
                current_time = time.time()
                if self.gello_listener is not None: # teleop mode
                    self.get_command()
                    command = [] if self.pause else list(self.command)

                else: # non-teleop mode
                    if self.key_states["r"]:
                        self.reset.value = True
                        self.log(f"keyboard robot reset: {self.reset.value}, resetting robot to default pose")
                        self.record_stop.value = True
                    if self.key_states["s"]:
                        self.reset.value = False
                        self.log(f"keyboard robot reset: {self.reset.value}")
                        self.record_start.value = True

                    command = list(self.command)

                command_np = np.array(self.command[:])
                assert command_np.shape[0] == 8, f"command shape should be (8,), got {command_np.shape}"                

                if self.bimanual:
                    self.command_sender_left.send([self.command[0][0:8]])
                    self.command_sender_right.send([self.command[0][8:16]])
                else:
                    self.command_sender.send([command]) # send command
                    # self.command_sender.send([[]]) # zero action for safety

                # data storage
                self.cur_time_q.value = current_time
                self.cur_qpos_comm[:] = command_np.tolist()

                fk = self.kin_helper.compute_fk_sapien_links(command_np[:7], [self.kin_helper.sapien_eef_idx])[0]
                self.cur_eef_trans[:] = fk.flatten()

                # print("command send freq: %.2f Hz"%(1.0 / (time.time() - current_time)))

            except Exception as e:
                print(f"Error in GelloTeleop", e.with_traceback())
                break
        
        self.stop()
        if self.bimanual:
            self.command_sender_left.close()
            self.command_sender_right.close()
        else:
            self.command_sender.close()

        if self.gello_listener is not None:
            self.gello_listener.stop()
        self.keyboard_listener.stop()
        # self.update_joints_t.join()
        print(f"{'='*20} keyboard + gello teleop exit!")
    
    @property
    def alive(self):
        alive = self._alive.value 
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