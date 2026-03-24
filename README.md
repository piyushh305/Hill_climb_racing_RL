# Hill_climb_racing_RL
# 🏔️ Hill Climb RL Agent

> A reinforcement learning agent that learns to play **Hill Climb Racing** autonomously on macOS using PPO (Proximal Policy Optimization) and real-time screen capture.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey?style=flat-square&logo=apple)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange?style=flat-square)

---

## 📌 Overview

This project trains an RL agent to autonomously control Hill Climb Racing on macOS. The agent observes the game screen in real-time, processes visual frames, and sends keyboard inputs to maximize distance and stability.

The agent is built on top of **Stable Baselines3's PPO** with a CNN policy, and uses screen capture + computer vision to extract state information directly from the game window — no game API or source code modification required.

---

## 🧠 How It Works

```
Game Screen → Screen Capture (mss) → Frame Processing (OpenCV)
     → State Extraction (angle, speed, slope, crash)
     → PPO Agent (CNN Policy)
     → Action (keyboard input via pynput)
     → Game responds → repeat
```

### State Representation
- **Visual Frame**: 84×84 grayscale screenshot of the game window
- **Angle**: Estimated body tilt via Hough line detection on edges
- **Speed**: Optical flow approximation using frame differencing
- **Slope**: Terrain gradient from the bottom region of the frame
- **Crash Detection**: Identifies freezes via frame history comparison

### Action Space (Discrete — 9 actions)
| Action | Keys |
|--------|------|
| 0 | No-op |
| 1 | ↑ Gas |
| 2 | ↓ Brake |
| 3 | → Right |
| 4 | ← Left |
| 5 | ↑ + → |
| 6 | ↑ + ← |
| 7 | ↓ + → |
| 8 | ↓ + ← |

### Reward Function
| Signal | Weight |
|--------|--------|
| Speed (motion proxy) | `+0.1 × speed` |
| Angle penalty | `-0.05 × \|angle\|` |
| Angular velocity penalty | `-0.02 × \|Δangle\|` |
| Gas usage penalty | `-0.01` |
| Brake usage penalty | `-0.05` |
| Stable driving bonus (60 frames) | `+5.0` |
| Crash penalty | `-100` |

---

## 📁 Project Structure

```
hill-climb-rl/
├── main.py           # Entry point — launches game, starts train or play
├── train.py          # PPO training loop with checkpointing
├── play.py           # Inference script for running trained model
├── env.py            # Gymnasium environment wrapper
├── vision.py         # Screen capture & state extraction (OpenCV + mss)
├── controller.py     # Keyboard input controller (pynput)
├── window.py         # macOS window detection via AppleScript
└── requirements.txt  # Python dependencies
```

---

## ⚙️ Requirements

### System
- macOS (tested on macOS 12+)
- Hill Climb Racing (Mac App Store)
- Python 3.8+

### macOS Permissions (**Required**)
Go to **System Settings → Privacy & Security** and enable both:
- ✅ **Accessibility** — for keyboard control (pynput)
- ✅ **Screen Recording** — for game capture (mss)

> ⚠️ Without these permissions, the agent cannot send inputs or capture the screen.

### Python Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
stable-baselines3[extra]
gymnasium
opencv-python
mss
pynput
numpy
```

**Optional — OCR-based start screen detection:**
```bash
brew install tesseract
pip install pytesseract
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/piyushh305/Hill_climb_racing_RL.git
cd Hill_climb_racing_RL
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Open Hill Climb Racing
Launch the game manually from your Applications or let `main.py` try to auto-open it.

### 4. Run the Agent
```bash
python3 main.py
```

The script will:
- Auto-detect the game window
- Prompt you to **train** a new model or **play** with an existing one
- Begin training for 100,000 timesteps (default), saving checkpoints every 50,000 steps

---

## 🏋️ Training

```bash
python3 train.py
```

Training runs PPO with a CNN policy for 100,000 timesteps by default (configurable). Checkpoints are saved to `./checkpoints/` every 50,000 steps. The final model is saved as `ppo_hillclimb.zip`.

### PPO Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | `2.5e-4` |
| n_steps | `2048` |
| Batch Size | `64` |
| n_epochs | `10` |
| Gamma | `0.99` |
| GAE Lambda | `0.95` |
| Clip Range | `0.25` |
| Entropy Coefficient | `0.001` |

---

## 🎮 Running the Trained Agent

```bash
python3 play.py
```

Loads `ppo_hillclimb.zip` and runs the agent for 5 episodes (configurable). Includes action smoothing to prevent sudden, erratic inputs.

---

## 🔍 Module Details

### `env.py` — Gymnasium Environment
- Wraps the game as a standard `gym.Env`
- Handles episode resets: only clicks CONTINUE/START when the car **actually crashes** (not on max-step truncation)
- Uses `frame_skip=3` and exponential action smoothing (`alpha=0.2`) for stable control
- Max steps per episode: `500`

### `vision.py` — Game Vision
- Captures frames using `mss` (fast screen capture)
- Extracts angle via Hough line transform on Canny edges
- Estimates speed via inter-frame pixel difference
- Detects crashes by monitoring frame history variance
- Optional: OCR-based start screen detection with Tesseract

### `window.py` — Window Detector
- Uses AppleScript via `osascript` to locate the Hill Climb Racing window
- Retrieves exact pixel coordinates and dimensions
- Falls back to a default region `(0, 25, 800, 600)` if detection fails

### `controller.py` — Input Controller
- Uses `pynput` for low-level keyboard simulation
- Cleanly releases all pressed keys before applying a new action
- Provides `release_all()` and `press_enter()` utilities

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `Could not press key` error | Grant **Accessibility** permission to Terminal |
| Black screen / no capture | Grant **Screen Recording** permission to Terminal |
| Window not found | Open Hill Climb Racing manually before running |
| Agent acts erratically | Ensure the game window is **fully visible** and **not minimized** |
| Crash detection too sensitive | Tune the `< 1.0` threshold in `vision.py → _detect_crash()` |

---

## 📈 Future Improvements

- [ ] Add reward shaping based on distance traveled (via OCR)
- [ ] Multi-frame stacking for better temporal awareness
- [ ] Curriculum learning — start on easier terrain
- [ ] Integrate TensorBoard for training visualization
- [ ] Support for Windows/Linux via platform-specific window detection

---

## 📄 License

This project is licensed under the open source See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL framework
- [Gymnasium](https://gymnasium.farama.org/) — Environment interface
- [mss](https://python-mss.readthedocs.io/) — Fast cross-platform screen capture
- [pynput](https://pynput.readthedocs.io/) — Keyboard & mouse control
- [OpenCV](https://opencv.org/) — Computer vision utilities

---

<div align="center">
  Made by <a href="https://github.com/piyushh305">piyushh305</a>
</div>
