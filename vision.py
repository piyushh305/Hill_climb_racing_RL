"""Frame capture and state extraction - Mac compatible"""
import mss
import numpy as np
import cv2
from collections import deque

# Optional: pytesseract for OCR-based start screen detection
# If not installed, falls back to image analysis only
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class GameVision:
    def __init__(self):
        self.sct = mss.mss()
        self.region = None
        self.prev_frame = None
        self.frame_history = deque(maxlen=3)
        self.start_keywords = ['start', 'play', 'tap to start', 'enter to start', 'press enter']

    def set_region(self, rect):
        """Set capture region from window rect."""
        self.region = {
            'left': rect['left'],
            'top': rect['top'],
            'width': rect['width'],
            'height': rect['height']
        }
        print(f"[Vision] Capture region set: {self.region}")

    def capture(self):
        """Capture frame from region."""
        if not self.region:
            print("[Vision] No region set. Call set_region() first.")
            return None
        try:
            img = np.array(self.sct.grab(self.region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[Vision] Capture error: {e}")
            return None

    def extract_state(self, frame):
        """Extract state features from frame."""
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (84, 84))

        # Angle proxy from edge orientation
        edges = cv2.Canny(gray, 50, 150)
        angle = self._compute_angle(edges)

        # Speed proxy from frame difference
        speed = self._compute_speed(gray)

        # Slope proxy from bottom region gradient
        slope = self._compute_slope(gray)

        # Crash detection
        crashed = self._detect_crash(gray)

        self.prev_frame = gray.copy()
        self.frame_history.append(small)

        return {
            'frame': small / 255.0,
            'angle': angle,
            'speed': speed,
            'slope': slope,
            'crashed': crashed
        }

    def _compute_angle(self, edges):
        """Estimate angle from edge orientation."""
        h, w = edges.shape
        roi = edges[h//3:2*h//3, w//3:2*w//3]
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)

        return float(np.median(angles)) if angles else 0.0

    def _compute_speed(self, gray):
        """Estimate speed from frame difference."""
        if self.prev_frame is None:
            return 0.0

        diff = cv2.absdiff(gray, self.prev_frame)
        motion = np.mean(diff) / 255.0
        return float(np.clip(motion * 10, 0, 1))

    def _compute_slope(self, gray):
        """Estimate terrain slope from bottom region."""
        h, w = gray.shape
        bottom = gray[int(h * 0.7):, :]

        grad_x = cv2.Sobel(bottom, cv2.CV_64F, 1, 0, ksize=3)
        slope = np.mean(grad_x) / 255.0
        return float(np.clip(slope, -1, 1))

    def _detect_crash(self, gray):
        """Detect crash by frame freeze."""
        if len(self.frame_history) < 3:
            return False

        diffs = []
        for i in range(len(self.frame_history) - 1):
            diff = np.mean(np.abs(
                self.frame_history[i].astype(float) -
                self.frame_history[i+1].astype(float)
            ))
            diffs.append(diff)

        return bool(np.mean(diffs) < 1.0) if diffs else False

    def detect_start_screen(self, frame):
        """Detect if start/play screen is visible."""
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Try OCR if tesseract is installed
        if TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(thresh).lower()
                for keyword in self.start_keywords:
                    if keyword in text:
                        return True
            except Exception:
                pass

        # Fallback: image analysis for high-contrast center (typical start buttons)
        h, w = gray.shape
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        std_dev = np.std(center_region)
        mean_val = np.mean(center_region)

        if std_dev > 50 and 100 < mean_val < 200:
            return True

        return False
