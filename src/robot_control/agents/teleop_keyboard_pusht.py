import threading
import multiprocess as mp
import time

import numpy as np
from pynput import keyboard

from .udp_util import udpReceiver, udpSender
from modules_teleop.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from modules_teleop.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX, POSITION_UPDATE_INTERVAL, COMMAND_CHECK_INTERVAL
from modules_teleop.kinematics_utils import KinHelper

np.set_printoptions(precision=2, suppress=True)

class PushtTeleop(mp.Process):
    
    XYZ_STEP = 10 # NOTE: unit mm
    XYZ_STEP_SLOW = 5
    
    name="pusht_teleop"
    kin_helper = KinHelper(robot_name='xarm7')

    def __init__(
        self, 
        teleop_mode: bool = True,  # True for teleop, False for policy rollouts
    ) -> None:
        super().__init__()

        self.key_states = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,

            ",": False,
            ".": False,
            "/": False, 

            "p": False,
            "r": False,  # Key for reset the position
            "space": False,  # Key for send current position
            "shift": False,  # Key for slow down the speed
        }

        # additional states
        self.init = True
        self.pause = False
        self.reset = mp.Value('b', False)
        self.record_start = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.record_failed = mp.Value('b', False)
        self.keyboard_listener = None

        # states for moving the arm
        self.cur_position = None
        self.cur_qpos = None
        self.command = mp.Array('d', [0.0] * 7)

        self.cur_comm_qpos_q = mp.Queue(maxsize=1)
        self.cur_comm_xy_q = mp.Queue(maxsize=1)

        self._alive = mp.Value('b', True)
        self.controller_quit = False
        self.teleop_mode = teleop_mode

        self.state_receiver = None # udpReceiver(ports={'xarm_state':XARM_STATE_PORT})
        self.command_sender = None # udpSender(port=XARM_CONTROL_PORT)

    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self, key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            if key == keyboard.Key.space:
                self.key_states["space"] = True
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = True

    def on_release(self, key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.space:
                self.key_states["space"] = False
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = False
            
            
            if key == keyboard.Key.esc:
                return False

    def update_xarm_pos(self):
        self.state_receiver = udpReceiver(ports={'xarm_state': XARM_STATE_PORT}, re_use_address=True)
        self.state_receiver.start()

        while self.alive:
            xarm_state = self.state_receiver.get("xarm_state", pop=True)
            # print(f"Received xarm state: {xarm_state}")
            if xarm_state is not None:
                self.cur_position = xarm_state["pos"]
                self.cur_qpos = xarm_state["qpos"]
                # print(f"Current position: {self.cur_position}, Current qpos: {self.cur_qpos}")
                # self.fk_trans_mat = xarm_state["e2b"]
            time.sleep(POSITION_UPDATE_INTERVAL / 10)

        self.state_receiver.stop()
        self.log(f"update_xarm_pos exit!")

    def get_command(self):
        if self.cur_position is None:
            # print("Current position is None, waiting for xarm state update...")
            return
            
        x, y, z, roll, pitch, yaw = self.cur_position
        # print(f" position before: {self.cur_position}")

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
            self.log(f"Record stop")
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
            # print("not paused, continue to get command")
            xyz_step = self.XYZ_STEP_SLOW if self.key_states["shift"] else self.XYZ_STEP
            if self.key_states["w"]:
                x += xyz_step
            if self.key_states["s"]:
                x -= xyz_step
            if self.key_states["a"]:
                y += xyz_step
            if self.key_states["d"]:
                y -= xyz_step

            controller_quit = False
            if self.key_states["space"]:
                # if space key is pressed, send current position
                controller_quit = True
            self.controller_quit = controller_quit

            if self.key_states["r"]:
                self.command = ["reset"]
                return
            
            # NOTE: hardcoded for pusht teleop
            z = 220.0
            roll = -178.0
            pitch = 0.0
            yaw = 1.3

            next_position = [x / 1000.0, y / 1000.0, z / 1000.0, roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi]
            # print(f"Next position: {next_position}")
            # if not self.cur_comm_xy_q.full():
                # self.cur_comm_xy_q.get()
            self.cur_comm_xy_q.put(np.array([x,y]) / 1000.0)

            # print(f"Current qpos: {self.cur_qpos}")
            qpos_goal = self.kin_helper.compute_ik_sapien(initial_qpos=self.cur_qpos, cartesian=np.array(next_position), verbose=False)
            # print(f"Computed qpos goal: {qpos_goal}")
            self.command[:] = qpos_goal

            return

    def run(self) -> None:
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        
        self.update_pos_t = threading.Thread(name="get_pos_from_XarmController", target=self.update_xarm_pos)
        self.update_pos_t.start()

        self.command_sender = udpSender(port=XARM_CONTROL_PORT)

        time.sleep(1)
        print("pusht teleop start!")

        while self.alive:
            try:
                if self.teleop_mode:
                    self.get_command()
                    # print(f"Current command: {self.command[:]}")
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
                # if not self.cur_comm_qpos_q.full():
                    # self.cur_comm_qpos_q.get()
                self.cur_comm_qpos_q.put(qpos)

                self.command_sender.send([command])
                # time.sleep(COMMAND_CHECK_INTERVAL / 2)
            except BaseException as e:
                self.log(f"Error in PushT Teleop: {e.with_traceback()}")
                break

        self.stop()
        self.command_sender.close()
        self.keyboard_listener.stop()
        self.update_pos_t.join()
        self.log(f"{'='*20} pusht teleop exit!")
    
    @property
    def alive(self):
        alive = self._alive.value & self.keyboard_listener.is_alive()
        self._alive.value = alive
        return alive 

    def stop(self, stop_controller=False):
        if stop_controller:
            self.log("teleop stop controller")
            self.command_sender.send(["quit"])
            time.sleep(1)
        self.log("teleop stop")
        self._alive.value = False


if __name__ == "__main__":
    pusht_teleop = PushtTeleop()
    pusht_teleop.start()
    pusht_teleop.join()