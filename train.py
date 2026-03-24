"""Training script"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import HillClimbEnv
import os


def train_agent(total_timesteps=1000000, model_path="ppo_hillclimb"):
    """Train PPO agent."""
    print("Initializing environment...")
    env = DummyVecEnv([lambda: HillClimbEnv()])

    print("Creating PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.001,
        tensorboard_log=None
    )

    os.makedirs("./checkpoints", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="ppo_hillclimb"
    )

    print("Starting training...")
    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    print(f"Saving model to {model_path}...")
    model.save(model_path)
    env.close()
    return model


if __name__ == "__main__":
    train_agent()
