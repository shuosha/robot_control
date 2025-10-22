# modules_teleop/aloha_teleop.py
import time
import multiprocess as mp
import numpy as np
from pynput import keyboard
from modules_teleop.kinematics_utils import KinHelper, pose6_to_matrix
import trossen_arm
import traceback

class AlohaTeleop(mp.Process):
    name = "aloha_teleop"
    kin_helper = KinHelper(robot_name='xarm7') # TODO: add aloha!

    SERVER_IP_LEADER_R = "192.168.1.2"
    SERVER_IP_LEADER_L = "192.168.1.3"
    SERVER_IP_FOLLOWER_R = "192.168.1.4"
    SERVER_IP_FOLLOWER_L = "192.168.1.5"
    FORCE_FEEDBACK_GAIN = 0.1

    def __init__(
        self,
        bimanual=True,
        teleop_mode=True,
    ):
        super().__init__()
        
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

        # command queues i.e., follower pose
        self.comm_qpos_R = mp.Array('d', [0.0] * 6)
        self.comm_trans_R = mp.Array('d', [0.0] * 16)
        self.comm_gripper_R = mp.Array('d', [0.0] * 1)

        self.comm_qpos_L = mp.Array('d', [0.0] * 6)
        self.comm_trans_L = mp.Array('d', [0.0] * 16)
        self.comm_gripper_L = mp.Array('d', [0.0] * 1)

        # robot queues i.e., leader pose
        self.cur_qpos_R = mp.Array('d', [0.0] * 6)
        self.cur_trans_R = mp.Array('d', [0.0] * 16)
        self.cur_gripper_R = mp.Array('d', [0.0] * 1)

        self.cur_qpos_L = mp.Array('d', [0.0] * 6)
        self.cur_trans_L = mp.Array('d', [0.0] * 16)
        self.cur_gripper_L = mp.Array('d', [0.0] * 1)

        self.cur_time_q = mp.Value('d', 0.0)

        self._alive = mp.Value('b', True)
        self.controller_quit = True
        self.initialize_done = mp.Value('b', False)

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
            
    def run(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        print("Keyboard listener start!")

        self._start_aloha_teleop()

        print("Aloha teleop start!")
        self.driver_leaderR.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver_leaderL.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver_followerR.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerL.set_all_modes(trossen_arm.Mode.position)
        while self.alive:
            try:
                current_time = time.time()
                leaderR_position = self.driver_leaderR.get_all_positions()
                leaderR_cart = self.driver_leaderR.get_cartesian_positions()
                leaderL_position = self.driver_leaderL.get_all_positions()
                leaderL_cart = self.driver_leaderL.get_cartesian_positions()

                followerR_position = self.driver_followerR.get_all_positions()
                followerL_position = self.driver_followerL.get_all_positions()
                followerR_cart = self.driver_followerR.get_cartesian_positions()
                followerL_cart = self.driver_followerL.get_cartesian_positions()

                # Feed the external efforts from the follower robot to the leader robot
                R_feedback = -self.FORCE_FEEDBACK_GAIN * np.array(self.driver_followerR.get_all_external_efforts())
                self.driver_leaderR.set_all_external_efforts(
                    R_feedback,
                    0.0,
                    False,
                )
                self.driver_followerR.set_all_positions(leaderR_position, 0.0, False)

                L_feedback = -self.FORCE_FEEDBACK_GAIN * np.array(self.driver_followerL.get_all_external_efforts())
                self.driver_leaderL.set_all_external_efforts(
                    L_feedback,
                    0.0,
                    False,
                )
                self.driver_followerL.set_all_positions(leaderL_position, 0.0, False)
                self.cur_time_q.value = current_time

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

                if self.key_states["s"]:
                    time.sleep(0.5)
                    break

                comm_qpos_R = np.array(leaderR_position)
                comm_cart_R = np.array(leaderR_cart)
                comm_trans_R = pose6_to_matrix(comm_cart_R)

                comm_qpos_L = np.array(leaderL_position)
                comm_cart_L = np.array(leaderL_cart)
                comm_trans_L = pose6_to_matrix(comm_cart_L)

                self.comm_qpos_R[:] = comm_qpos_R[:-1] # (6, )
                self.comm_gripper_R[:] = comm_qpos_R[-1:] # (1, )
                self.comm_trans_R[:] = comm_trans_R.flatten()

                self.comm_qpos_L[:] = comm_qpos_L[:-1] # (6, )
                self.comm_gripper_L[:] = comm_qpos_L[-1:] # (1, )
                self.comm_trans_L[:] = comm_trans_L.flatten()

                qpos_R = np.array(followerR_position)
                cart_R = np.array(followerR_cart)
                trans_R = pose6_to_matrix(cart_R)

                qpos_L = np.array(followerL_position)
                cart_L = np.array(followerL_cart)
                trans_L = pose6_to_matrix(cart_L)

                self.cur_qpos_R[:] = qpos_R[:-1]
                self.cur_gripper_R[:] = qpos_R[-1:]
                self.cur_trans_R[:] = trans_R.flatten()

                self.cur_qpos_L[:] = qpos_L[:-1]
                self.cur_gripper_L[:] = qpos_L[-1:]
                self.cur_trans_L[:] = trans_L.flatten()

                end_time = time.time()
                time.sleep(max(0.01 - end_time + current_time, 0.0))

                if not self.initialize_done.value:
                    self.initialize_done.value = True
                    print("Initialization complete")
                
            except Exception as e:
                print(f"Error in teleop: {e}")
                traceback.print_exc()
                break

        self.stop()
        self.keyboard_listener.stop()
        print(f"{'='*20} Aloha teleop stopped!")


    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    @property
    def alive(self):
        alive = self._alive.value 
        self._alive.value = alive
        return alive 
    
    def stop(self, stop_controller=False):
        if stop_controller:
            self.log("teleop stop controller")
            time.sleep(1)
        self._alive.value = False
        self._stop_aloha_teleop()
        self.log("teleop stop")

    def _start_aloha_teleop(self):
        print("Initializing the drivers...")
        self.driver_leaderR = trossen_arm.TrossenArmDriver()
        self.driver_leaderL = trossen_arm.TrossenArmDriver()
        self.driver_followerR = trossen_arm.TrossenArmDriver()
        self.driver_followerL = trossen_arm.TrossenArmDriver()

        print("Configuring the drivers...")
        self.driver_leaderR.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            self.SERVER_IP_LEADER_R,
            False,
        )
        self.driver_leaderL.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            self.SERVER_IP_LEADER_L,
            False,
        )

        self.driver_followerR.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            self.SERVER_IP_FOLLOWER_R,
            False,
        )
        self.driver_followerL.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            self.SERVER_IP_FOLLOWER_L,
            False,
        )

        print("Moving to home positions...")
        self.driver_leaderR.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderR.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )
        self.driver_followerR.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerR.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )

        self.driver_leaderL.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderL.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )
        self.driver_followerL.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerL.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )

        print("Starting to teleoperate the robots...")
        time.sleep(1)


    def _stop_aloha_teleop(self):
        print("Moving to home positions...")
        self.driver_leaderR.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderR.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )
        self.driver_followerR.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerR.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )

        self.driver_leaderL.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderL.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )
        self.driver_followerL.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerL.set_all_positions(
            np.array([0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
            2.0,
            True,
        )

        print("Moving to sleep positions...")
        self.driver_leaderR.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderR.set_all_positions(np.zeros(self.driver_leaderR.get_num_joints()), 2.0, True)
        self.driver_followerR.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerR.set_all_positions(np.zeros(self.driver_followerR.get_num_joints()), 2.0, True)

        self.driver_leaderL.set_all_modes(trossen_arm.Mode.position)
        self.driver_leaderL.set_all_positions(np.zeros(self.driver_leaderL.get_num_joints()), 2.0, True)
        self.driver_followerL.set_all_modes(trossen_arm.Mode.position)
        self.driver_followerL.set_all_positions(np.zeros(self.driver_followerL.get_num_joints()), 2.0, True)