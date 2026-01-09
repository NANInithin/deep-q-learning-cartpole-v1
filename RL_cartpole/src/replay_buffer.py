import random
from collections import deque
from typing import Deque, Tuple, List, Optional

import numpy as np


class ReplayBuffer:
    """Experience replay buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if len(self.buffer) < batch_size:
            return None

        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(
            self.buffer, batch_size
        )
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
