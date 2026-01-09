from collections import deque
from typing import List

import gymnasium as gym
import numpy as np
import torch

from .agent import DQNAgent, DEVICE


SEED = 42
SOLVED_SCORE = 195.0


def train_dqn_agent(
    env_name: str = "CartPole-v1",
    n_episodes: int = 2000,
    target_update_freq: int = 100,
) -> List[float]:
    env = gym.make(env_name)
    env.reset(seed=SEED)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        learning_rate=1e-4,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        buffer_capacity=10000,
        batch_size=64,
    )  # matches your notebook config.[file:2]

    scores: List[float] = []
    scores_window = deque(maxlen=100)

    print(f"--- Starting Training on {env_name} using {DEVICE} ---")
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=SEED if i_episode == 1 else None)
        score = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            score += reward

        scores_window.append(score)
        scores.append(score)

        if i_episode % target_update_freq == 0:
            agent.update_target_net()

        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode:4d} | "
                f"Avg Score: {np.mean(scores_window):6.2f} | "
                f"Epsilon: {agent.epsilon:6.4f}"
            )

        if np.mean(scores_window) >= SOLVED_SCORE and i_episode >= 100:
            print(
                f"Environment solved in {i_episode - 100:d} episodes! "
                f"Average Score: {np.mean(scores_window):.2f}"
            )
            break

    env.close()
    return scores, agent


if __name__ == "__main__":
    scores, agent = train_dqn_agent()

    torch.save(agent.policy_net.state_dict(), "RL_cartpole/models/cartpole_dqn_weights.pth")
    print("✅ Model saved to RL_cartpole/models/cartpole_dqn_weights.pth")
