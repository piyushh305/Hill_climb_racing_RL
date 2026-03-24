"""Window detection - Mac compatible"""
import subprocess
import time


class WindowDetector:
    def __init__(self, window_keywords=None):
        self.window_keywords = window_keywords or ["hill", "climb"]
        self.rect = None
        self._app_name = "Hill Climb Racing"

    def find_window(self):
        try:
            script = '''
            tell application "System Events"
                return exists process "Hill Climb Racing"
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                  capture_output=True, text=True, timeout=10)
            if "true" in result.stdout.lower():
                self._update_rect()
                return True
        except Exception as e:
            print(f"Window search error: {e}")
        return False

    def _update_rect(self):
        try:
            script = '''
            tell application "System Events"
                tell process "Hill Climb Racing"
                    set pos to position of window 1
                    set sz to size of window 1
                    return ((item 1 of pos) as string) & "," & ((item 2 of pos) as string) & "," & ((item 1 of sz) as string) & "," & ((item 2 of sz) as string)
                end tell
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                  capture_output=True, text=True, timeout=5)
            parts = result.stdout.strip().split(",")
            if len(parts) == 4:
                self.rect = {
                    'left': int(float(parts[0])),
                    'top': int(float(parts[1])),
                    'width': int(float(parts[2])),
                    'height': int(float(parts[3]))
                }
                print(f"Window region: {self.rect}")
                return
        except Exception as e:
            print(f"Could not get window rect: {e}")
        self.rect = {'left': 0, 'top': 25, 'width': 800, 'height': 600}
        print(f"Using default region: {self.rect}")

    def update_rect(self):
        self._update_rect()

    def wait_for_window(self, timeout=60):
        print(f"Searching for Hill Climb Racing...")
        start = time.time()
        while time.time() - start < timeout:
            if self.find_window():
                print(f"Found window! Region: {self.rect}")
                return True
            print("Window not found, retrying in 2s...")
            time.sleep(2)
        return False

    def is_active(self):
        try:
            script = '''
            tell application "System Events"
                return exists process "Hill Climb Racing"
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                  capture_output=True, text=True, timeout=5)
            return "true" in result.stdout.lower()
        except:
            return False

    def focus(self):
        try:
            subprocess.run(['osascript', '-e',
                          'tell application "Hill Climb Racing" to activate'],
                         timeout=5)
            time.sleep(0.3)
        except Exception as e:
            print(f"Could not focus window: {e}")
