"""Main entry point - Mac compatible"""
import os
import sys
import time
import subprocess
from train import train_agent
from play import play_agent
from window import WindowDetector
from vision import GameVision
from controller import GameController


def auto_start_game():
    """Detect game window and prepare for training."""
    window = WindowDetector()
    vision = GameVision()
    controller = GameController()

    # Try to open the game if not already running
    print("Looking for Hill Climb Racing...")
    try:
        subprocess.run(
            ['open', '-a', 'Hill Climb Racing'],
            capture_output=True, timeout=5
        )
        time.sleep(3)
    except Exception:
        print("Could not auto-launch game. Please open Hill Climb Racing manually.")

    if not window.wait_for_window(timeout=30):
        print("\n❌ Game window not found after 30 seconds.")
        print("Please open Hill Climb Racing and try again.")
        return False

    vision.set_region(window.rect)
    print("✅ Game window detected!")

    # Focus and attempt to dismiss start screen
    window.focus()
    time.sleep(1)

    max_attempts = 10
    for attempt in range(max_attempts):
        frame = vision.capture()
        if vision.detect_start_screen(frame):
            print(f"Start screen detected (attempt {attempt + 1}), pressing Enter...")
            controller.press_enter()
            time.sleep(1)
        else:
            print("Game is ready!")
            break
        time.sleep(1)

    return True


def main():
    model_path = "ppo_hillclimb"

    print("=" * 60)
    print("  Hill Climb RL Agent 🤖⛰️  (Mac Version)")
    print("=" * 60)

    print("\n🔐 Mac Permissions Reminder:")
    print("   Terminal needs Accessibility + Screen Recording permissions.")
    print("   System Settings → Privacy & Security → enable both for Terminal\n")

    print("Auto-starting game...")
    if not auto_start_game():
        print("Failed to start game automatically.")
        return

    if os.path.exists(f"{model_path}.zip"):
        print("\n✅ Trained model found!")
        choice = input("Train new model (t) or Play existing (p)? [p]: ").strip().lower()

        if choice == 't':
            print("\nStarting training (1M timesteps)...")
            train_agent(total_timesteps=100000, model_path=model_path)
        else:
            print("\nRunning trained agent...")
            play_agent(model_path=model_path, num_episodes=5)
    else:
        print("\n⚠️  No trained model found. Starting training from scratch...")
        print("This will take 2-4 hours. Keep Hill Climb Racing visible on screen.\n")
        train_agent(total_timesteps=100000, model_path=model_path)

        print("\n✅ Training complete! Starting demo...")
        play_agent(model_path=model_path, num_episodes=3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
