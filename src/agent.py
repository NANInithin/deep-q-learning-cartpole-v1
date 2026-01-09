import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import QNetwork
from .replay_buffer import ReplayBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """DQN agent with target network and experience replay for CartPole-v1."""

    def __init__(self, state_size: int, action_size: int, **kwargs) -> None:
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters (defaults taken from your notebook).[file:2]
        self.gamma = kwargs.get("gamma", 0.99)
        self.learning_rate = kwargs.get("learning_rate", 5e-4)
        self.batch_size = kwargs.get("batch_size", 64)

        # Epsilon-greedy
        self.epsilon = kwargs.get("epsilon_start", 1.0)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)

        # Networks
        self.policy_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        buffer_capacity = kwargs.get("buffer_capacity", 10000)
        self.memory = ReplayBuffer(buffer_capacity)

    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> None:
        """Sample from replay buffer and update Q-network."""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch

        states_t = torch.from_numpy(states).float().to(DEVICE)
        actions_t = torch.from_numpy(actions).long().to(DEVICE)
        rewards_t = torch.from_numpy(rewards).float().to(DEVICE)
        next_states_t = torch.from_numpy(next_states).float().to(DEVICE)
        dones_t = torch.from_numpy(dones).float().to(DEVICE)

        # Q(s, a)
        q_expected = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        q_next = self.target_net(next_states_t).detach().max(1)[0]

        # Target: r + gamma * max_a' Q(s', a') * (1 - done)
        q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.criterion(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
