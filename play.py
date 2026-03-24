"""Inference script"""
from stable_baselines3 import PPO
from env import HillClimbEnv
import time
import numpy as np


def play_agent(model_path="ppo_hillclimb", num_episodes=5):
    """Run trained agent with smooth, stable control."""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Initializing environment...")
    env = HillClimbEnv()

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        prev_action = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Clamp action to avoid sudden flips
            if abs(action - prev_action) > 3:
                action = prev_action + np.sign(action - prev_action) * 2

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            prev_action = action

            if steps % 30 == 0:
                print(f"Step {steps} | Reward: {total_reward:.2f}")

        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        time.sleep(2)

    env.close()


if __name__ == "__main__":
    play_agent()
