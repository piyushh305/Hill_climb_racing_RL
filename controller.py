"""Keyboard controller - Mac compatible via pynput"""
from pynput.keyboard import Controller, Key
import time


class GameController:
    def __init__(self):
        self.kb = Controller()
        self.action_map = {
            0: [],
            1: [Key.up],
            2: [Key.down],
            3: [Key.right],
            4: [Key.left],
            5: [Key.up, Key.right],
            6: [Key.up, Key.left],
            7: [Key.down, Key.right],
            8: [Key.down, Key.left]
        }
        self.pressed = set()

    def execute(self, action):
        """Execute discrete action."""
        keys = self.action_map.get(action, [])

        # Release all currently pressed keys
        for key in list(self.pressed):
            try:
                self.kb.release(key)
            except Exception:
                pass
        self.pressed.clear()

        # Press new keys
        for key in keys:
            try:
                self.kb.press(key)
                self.pressed.add(key)
            except Exception as e:
                print(f"[Controller] Could not press key {key}: {e}")
                print("  -> Make sure Terminal has Accessibility permission in:")
                print("     System Settings → Privacy & Security → Accessibility")

    def release_all(self):
        """Release all keys."""
        for key in list(self.pressed):
            try:
                self.kb.release(key)
            except Exception:
                pass
        self.pressed.clear()

    def press_enter(self):
        """Press Enter key to start game."""
        try:
            self.kb.press(Key.enter)
            time.sleep(0.1)
            self.kb.release(Key.enter)
        except Exception as e:
            print(f"[Controller] Could not press Enter: {e}")
