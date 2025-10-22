import threading
import multiprocess as mp
import time

import numpy as np
from pynput import keyboard

# from .udp_util import udpReceiver, udpSender
# from modules_teleop.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT
from modules_teleop.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX, POSITION_UPDATE_INTERVAL, COMMAND_CHECK_INTERVAL

from multiprocess.managers import SharedMemoryManager
from camera.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty

np.set_printoptions(precision=2, suppress=True)

class KeyboardTeleopNoRobot(mp.Process):
    
    XYZ_STEP = 5
    XYZ_STEP_STOW = 1
    ANGLE_STEP = 1
    ANGLE_STEP_STOW = 0.2
    GRIPPER_STEP = 100
    GRIPPER_STEP_STOW = 20
    
    name = "keyboard_teleop_no_robot"

    def __init__(
        self, 
        gripper_enable=False,
    ) -> None:
        super().__init__()
        self.gripper_enable = gripper_enable
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

        self.key_states = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "up": False,    # Up arrow key for moving up
            "down": False,  # Down arrow key for moving down
            "left": False,
            "right": False,
            
            "i": False,
            "k": False,
            "j": False,
            "l": False,
            "u": False,
            "o": False,

            "[": False,
            "]": False,
            "{": False,
            "}": False,

            ",": False,
            ".": False,

            "p": False,
            "r": False,  # Key for reset the position
            "space": False,  # Key for send current position
            "shift": False,  # Key for slow down the speed
        }

        # additional states
        self.init = True
        self.pause = False
        self.initialize_done = mp.Value('b', True)

        # states for moving the arm
        self.cur_position = np.zeros(6)
        self.command = []

        self._alive = mp.Value('b', True)

        shm_manager = SharedMemoryManager()
        shm_manager.start()
        examples = {
            'command': np.zeros(7),
            'timestamp': 0.0
        }
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=10240,
        )
        self.command_queue = command_queue

    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self, key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = True
            elif key == keyboard.Key.down:
                self.key_states["down"] = True
            elif key == keyboard.Key.left:
                self.key_states["left"] = True
            elif key == keyboard.Key.right:
                self.key_states["right"] = True
            elif key == keyboard.Key.space:
                self.key_states["space"] = True
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = True

    def on_release(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = False
            elif key == keyboard.Key.down:
                self.key_states["down"] = False
            elif key == keyboard.Key.left:
                self.key_states["left"] = False
            elif key == keyboard.Key.right:
                self.key_states["right"] = False
            elif key == keyboard.Key.space:
                self.key_states["space"] = False
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = False
            
            
            if key == keyboard.Key.esc:
                return False

    def get_command(self):
        if self.cur_position is None:
            return
            
        x, y, z, roll, pitch, yaw = self.cur_position
        
        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"keyboard teleop pause: {self.pause}")
            time.sleep(0.5)

        if self.pause:
            self.command = []
            return
        else:
            xyz_step = self.XYZ_STEP_STOW if self.key_states["shift"] else self.XYZ_STEP
            angle_step = self.ANGLE_STEP_STOW if self.key_states["shift"] else self.ANGLE_STEP
            gripper_step = self.GRIPPER_STEP_STOW if self.key_states["shift"] else self.GRIPPER_STEP
            gripper_changed = False
            if self.key_states["w"]:
                x += xyz_step
            if self.key_states["s"]:
                x -= xyz_step
            if self.key_states["a"]:
                y += xyz_step
            if self.key_states["d"]:
                y -= xyz_step
            if self.key_states["up"]:
                z += xyz_step   # Move up
            if self.key_states["down"]:
                z -= xyz_step   # Move down
            
            # adjusting the camera orientation
            if self.key_states["k"]:
                pitch = (pitch + angle_step) % 360
            if self.key_states["i"]:
                pitch = (pitch - angle_step) % 360
            if self.key_states["u"]:
                yaw = (yaw + angle_step) % 360
            if self.key_states["o"]:
                yaw = (yaw - angle_step) % 360
            if self.key_states["j"]:
                roll = (roll + angle_step) % 360
            if self.key_states["l"]:
                roll = (roll - angle_step) % 360

            if self.key_states["["]:
                gripper_pos -= gripper_step
                gripper_changed = True
            if self.key_states["]"]:
                gripper_pos += gripper_step
                gripper_changed = True
            if self.key_states["{"]:
                gripper_pos -= gripper_step
                gripper_changed = True
            if self.key_states["}"]:
                gripper_pos += gripper_step
                gripper_changed = True
            if self.key_states["left"]:
                gripper_pos = GRIPPER_OPEN_MIN
                gripper_changed = True
            elif self.key_states["right"]:
                gripper_pos = GRIPPER_OPEN_MAX
                gripper_changed = True

            controller_quit = False
            if self.key_states["space"]:
                # if space key is pressed, send current position
                controller_quit = True
            self.controller_quit = controller_quit

            if self.key_states["r"]:
                self.command = ["reset"]
                return

            if self.gripper_enable and gripper_changed:
                next_position = [x, y, z, roll, pitch, yaw, gripper_pos]
            else:
                next_position = [x, y, z, roll, pitch, yaw]
            
            if any([self.key_states[x] for x in ["w", "s", "a", "d", "up", "down",
                                                 "i", "k", "j", "l", "u", "o",
                                                 "[", "]", "{", "}", "left", "right", 
                                                 "space"]]):
                self.command = [next_position]
            else:
                self.command = []
            return
    
    def init_cur_position(self, cur_position): 
        self.cur_position = cur_position

    def run(self) -> None:
        self.log(f"{'='*20} keyboard teleop start!")
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        time.sleep(1)

        while self.alive:
            try:
                self.get_command()
                if self.command == []:
                    command_put = np.zeros(7)
                else:
                    assert len(self.command) == 1
                    command_put = np.array(self.command[0])
                    # print(command_put.shape)
                    if len(command_put) == 6:
                        command_put = np.append(command_put, 0)
                
                put_data = {
                    "command": command_put,
                    "timestamp": time.time()
                }
                # print(self.alive, put_data)
                self.command_queue.put(put_data)
                time.sleep(COMMAND_CHECK_INTERVAL / 2)
            except:
                self.log("Error in keyboard teleop")
                break

        self.keyboard_listener.stop()
        self.stop()
        self.log(f"{'='*20} keyboard teleop exit!")
    
    def get(self):
        if not self.command_queue:
            self.log("command queue is not initialized")
            return None
        try:
            commands = self.command_queue.get_all()
            self.command_queue.clear()
            return commands
        except Empty:
            # self.log("command queue is empty")
            return None

    @property
    def alive(self):
        alive = self._alive.value & self.keyboard_listener.is_alive()
        self._alive.value = alive
        return alive
    
    def start(self):
        self.log("teleop start")
        super().start()
        self._alive.value = True

    def stop(self):
        self.log("teleop stop")
        self._alive.value = False


if __name__ == "__main__":
    keyboard_teleop = KeyboardTeleopNoRobot()
    keyboard_teleop.start()
    keyboard_teleop.join()

