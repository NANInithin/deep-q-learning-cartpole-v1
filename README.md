# DQN CartPole-v1: Reinforcement Learning TP1 🚀

[![Training Results](cartpole_agent_performance.gif)](cartpole_agent_performance.gif)

**Deep Q-Network (DQN) implementation solving Gymnasium's CartPole-v1 environment.** This project implements a standard DQN agent with Experience Replay and a Target Network to stabilize training.

* **Status:** ✅ Solved
* **Performance:** Achieved **195.16 average score over 100 episodes** in **448 episodes**.
* **Evaluation:** Score **262/500** (pure greedy policy).

**Author:** Kopparapu Nithin Sai Kumar | Paris-Saclay University | Machine Vision & AI MSc

---

## 🎯 Project Objective

The goal is to solve the **CartPole-v1** environment from Gymnasium.

* **Goal:** Balance the pole by moving the cart left or right.
* **Solved Criteria:** Achieve an average reward of **≥ 195.0** over 100 consecutive episodes.
* **Max Steps:** 500 timesteps per episode.

---

## 🏗️ CartPole-v1 Environment

| Feature | Details |
| :--- | :--- |
| **State Space** | 4D Continuous: `[cart_pos, cart_vel, pole_angle, pole_ang_vel]` |
| **Action Space** | Discrete (2): `0=Left`, `1=Right` |
| **Reward** | +1 per timestep the pole stays balanced |
| **Fail Condition** | Pole Angle > 12° **OR** Cart Position > 2.4 |

---

## 🧠 Architecture & Algorithm

### Neural Network (Q-Function)
A simple Feed-Forward Neural Network (MLP) approximates the Q-values.

| Layer | Neurons | Activation |
| :--- | :--- | :--- |
| Input | 4 (State) | - |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 2 (Actions) | Linear |

### Key DQN Components
1.  **Experience Replay:** A buffer (Capacity: 10,000) stores transitions `(s, a, r, s', d)` to break correlations in training data.
2.  **Target Network:** A separate network calculates the target Q-values to prevent moving target instability. It is updated (hard sync) every **100 episodes**.
3.  **Loss Function:** Mean Squared Error (MSE) between the prediction and the Bellman target:
    $$L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

### Hyperparameters
These parameters were optimized experimentally:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **γ (Gamma)** | 0.99 | Discount factor for future rewards |
| **Learning Rate** | 1e-4 | Adam Optimizer learning rate |
| **ε_start** | 1.0 | Initial exploration rate |
| **ε_min** | 0.05 | Minimum exploration floor |
| **ε_decay** | 0.999 | Multiplicative decay per episode |
| **Batch Size** | 64 | Samples per training step |
| **Target Update** | 100 | Episodes between target net syncs |

---

## 📂 Project Structure

```text
RL_cartpole/
├── src/                      # Modular Python implementation
│   ├── __init__.py
│   ├── agent.py              # DQNAgent class (Target net + Replay logic)
│   ├── models.py             # PyTorch QNetwork (4-64-64-2)
│   ├── replay_buffer.py      # Experience Replay Buffer (deque)
│   ├── train.py              # Main training loop + early stopping
│   └── evaluate.py           # Evaluation & GIF generation
├── models/                   # Saved model weights
│   └── cartpole_dqn_weights.pth
├── cartpole_agent_performance.gif  # Visualization
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
