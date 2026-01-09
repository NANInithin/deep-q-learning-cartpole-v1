from typing import List

import gymnasium as gym
import imageio.v3 as iio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import DQNAgent, DEVICE
from .models import QNetwork


def evaluate_and_capture_frames(
    env_name: str,
    model_path: str,
    output_gif: str = "cartpole_agent_performance.gif",
) -> List[np.ndarray]:
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.policy_net.eval()
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return []

    frames: List[np.ndarray] = []
    done = False
    truncated = False
    score = 0.0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation.[file:2]

    print("Running evaluation episode...")
    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)

        state = next_state
        score += reward

    env.close()
    agent.epsilon = original_epsilon
    print(f"Evaluation finished. Score: {score}")
    print(f"Captured {len(frames)} frames.")

    iio.imwrite(output_gif, frames, duration=0.02, loop=0)
    print(f"Animation saved as {output_gif}")

    # Optional: show first frame
    img = mpimg.imread(output_gif)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    return frames


if __name__ == "__main__":
    evaluate_and_capture_frames(
        env_name="CartPole-v1",
        model_path="RL_cartpole/models/cartpole_dqn_weights.pth",
        output_gif="cartpole_agent_performance.gif",
    )
