"""Gym environment wrapper - Mac compatible"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from window import WindowDetector
from vision import GameVision
from controller import GameController
from pynput.mouse import Button, Controller as MouseController
import subprocess
import time
from collections import deque


class HillClimbEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.window = WindowDetector()
        self.vision = GameVision()
        self.controller = GameController()
        self.mouse = MouseController()

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )

        self.continue_x = 1019
        self.continue_y = 779
        self.start_x = 1038
        self.start_y = 785

        self.max_steps = 500
        self.step_count = 0
        self.prev_action = 0
        self.action_alpha = 0.2
        self.frame_skip = 3
        self.frames_since_action = 0
        self.current_action = 0
        self.angle_history = deque(maxlen=60)
        self.stable_angle_frames = 0
        self.prev_angle = 0
        self.prev_speed = 0
        self._window_initialized = False
        self._crashed = False  # Only True when car actually crashed

    def _init_window(self):
        if not self._window_initialized:
            print("Initializing game window (once)...")
            if not self.window.wait_for_window(timeout=30):
                raise RuntimeError("Game window not found.")
            self.vision.set_region(self.window.rect)
            self._window_initialized = True
            print("Window initialized!")

    def _focus_game(self):
        subprocess.run(
            ['osascript', '-e', 'tell application "Hill Climb Racing" to activate'],
            capture_output=True
        )
        time.sleep(0.3)

    def _click(self, x, y):
        self.mouse.position = (x, y)
        time.sleep(0.2)
        self.mouse.click(Button.left, 1)
        time.sleep(0.5)

    def _handle_crash(self):
        """Only called when car actually crashes or runs out of fuel."""
        print("Car crashed/out of fuel - clicking CONTINUE then START...")
        self._focus_game()
        time.sleep(0.8)
        self._click(self.continue_x, self.continue_y)
        time.sleep(1.5)
        self._click(self.start_x, self.start_y)
        time.sleep(1.5)
        print("New race started!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_window()
        self.controller.release_all()

        # ONLY click if car crashed - NOT if max steps was reached
        if self._crashed:
            self._handle_crash()
            self._crashed = False

        time.sleep(0.3)
        self.step_count = 0
        self.prev_action = 0
        self.current_action = 0
        self.frames_since_action = 0
        self.angle_history.clear()
        self.stable_angle_frames = 0
        self.prev_angle = 0
        self.prev_speed = 0

        frame = self.vision.capture()
        state = self.vision.extract_state(frame)
        if state is None:
            obs = np.zeros((1, 84, 84), dtype=np.uint8)
        else:
            obs = (state['frame'] * 255).astype(np.uint8).reshape(1, 84, 84)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.frames_since_action += 1

        if self.frames_since_action >= self.frame_skip:
            smoothed = int(self.action_alpha * action + (1 - self.action_alpha) * self.prev_action)
            self.current_action = smoothed
            self.prev_action = smoothed
            self.frames_since_action = 0

        self.controller.execute(self.current_action)
        time.sleep(0.005)

        frame = self.vision.capture()
        state = self.vision.extract_state(frame)

        if state is None:
            obs = np.zeros((1, 84, 84), dtype=np.uint8)
            self._crashed = True
            return obs, -100, True, False, {}

        obs = (state['frame'] * 255).astype(np.uint8).reshape(1, 84, 84)
        angle = state['angle']
        speed = state['speed']
        angular_velocity = angle - self.prev_angle

        if abs(angle) <= 10:
            self.stable_angle_frames += 1
        else:
            self.stable_angle_frames = 0

        reward = 0
        reward += speed * 0.1
        reward -= abs(angle) * 0.05
        reward -= abs(angular_velocity) * 0.02
        if self.current_action in [1, 5, 6]:
            reward -= 0.01
        if self.current_action in [2, 7, 8]:
            reward -= 0.05
        if self.stable_angle_frames >= 60:
            reward += 5.0
            self.stable_angle_frames = 0

        # Only set crashed flag if car actually crashed
        terminated = state['crashed']
        if terminated:
            self._crashed = True
            reward = -100

        # Truncated = max steps reached, game still running, NO clicking needed
        truncated = self.step_count >= self.max_steps

        self.prev_angle = angle
        self.prev_speed = speed

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.controller.release_all()
