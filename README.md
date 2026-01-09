# DQN for CartPole-v1 🚀

Deep Q-Network implementation solving Gymnasium's CartPole-v1 environment. Achieved **195+ average score over 100 episodes** in **448 episodes**.[file:2]

## Results
![Learning Curve](cartpole_agent_performance.gif)

**Score**: 262/500 timesteps (pure greedy policy)

## Features
- Target network + experience replay (10k buffer)
- Epsilon-greedy (1.0 → 0.05, decay=0.999)
- Two hidden layers (64 units, ReLU), Adam optimizer

## Quick Start
```bash
pip install -r requirements.txt
python -m src.train          # Train (~5-10 min)
python -m src.evaluate       # Generate GIF
