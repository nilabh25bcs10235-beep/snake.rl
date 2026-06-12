# Snake RL

A pure self-learning Deep Q-Network (DQN) agent that learns to play Snake from scratch using reinforcement learning. No pathfinding hints or pre-programmed strategy — every move comes from the neural network.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install pygame torch numpy matplotlib
```

## Train

Headless training is strongly recommended (much faster than visual mode):

```bash
python train.py --games 500
python train.py --target-score 55
python train.py --visual     # slower, shows pygame window
```

Models are saved to `model/`:
- `model.pth` — best rolling 100-game average
- `model_best_score.pth` — best single-game score

Training progress is written to `progress.png`.

## Evaluate

```bash
python evaluate.py --visual
python evaluate.py --model ./model/model_best_score.pth --games 10
```

## Stack

- Python 3.11+
- PyTorch
- Pygame
- DQN with experience replay and target network

## How it works

- **State**: 11 binary features (danger sensors, direction, food position)
- **Actions**: straight, turn right, turn left
- **Rewards**: +10 for food, -10 for death
- **Training**: epsilon-greedy exploration decaying over games, replay buffer, batch learning