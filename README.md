# Snake RL

A pure self-learning [Deep Q-Network (DQN)](https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning) agent that learns to play Snake from scratch. No pathfinding hints or pre-programmed strategy — every move comes from the neural network.

**Repository:** [github.com/nilabh25bcs10235-beep/snake.rl](https://github.com/nilabh25bcs10235-beep/snake.rl)

---

## Quick start (localhost)

The easiest way to run the project is the built-in local launcher:

```bash
git clone https://github.com/nilabh25bcs10235-beep/snake.rl.git
cd snake.rl
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements-ml.txt
python serve.py
```

Then open **[http://localhost:8080](http://localhost:8080)** in your browser.

### Deploy on Vercel (static site)

Vercel hosts the **landing page** (`index.html`) with run instructions. No Python serverless — avoids Vercel function errors. The pygame game always runs **on your machine**.

1. Import repo on [vercel.com](https://vercel.com)
2. Set **Framework Preset** → **Other**
3. Deploy (uses `vercel.json` automatically)

ML dependencies stay in `requirements-ml.txt` for local use only.

| Button | What it does |
|--------|----------------|
| **Play** | Opens the pygame window and watches the trained AI play |
| **Train** | Starts 100 headless training games in a new terminal |

**Windows shortcut:** double-click `run.bat` — it sets up the venv and opens the launcher.

---

## Setup (manual)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-ml.txt
```

Dependencies: `pygame`, `torch`, `numpy`, `matplotlib` (see [requirements-ml.txt](requirements-ml.txt)).

---

## Train

Headless training is strongly recommended (much faster than visual mode):

```bash
python train.py --games 500
python train.py --target-score 55
python train.py --visual         # slower — shows pygame window
```

Saved models (in `model/`):

| File | Description |
|------|-------------|
| `model.pth` | Best rolling 100-game average |
| `model_best_score.pth` | Best single-game score |

Training progress is saved to `progress.png`.

**Tip:** expect scores around **30–55+** only after **500–1000+** headless games. Pure RL takes time.

---

## Evaluate

```bash
python evaluate.py --visual
python evaluate.py --model ./model/model_best_score.pth --games 10
```

Close the pygame window to stop visual evaluation.

---

## Project structure

```
snake.rl/
├── index.html      # Vercel static landing page
├── vercel.json     # Static deploy config (no Python functions)
├── serve.py        # Local launcher → http://localhost:8080
├── train.py        # Train the DQN agent
├── evaluate.py     # Run a saved model
├── game.py         # Snake environment + pygame UI
├── agent.py        # DQN agent + exploration
├── model.py        # Neural network + trainer
├── run.bat         # Windows one-click launcher
└── requirements-ml.txt # Local ML deps (pygame, torch, …)
```

---

## How it works

| Component | Details |
|-----------|---------|
| **State** | 11 binary features (danger sensors, direction, food position) |
| **Actions** | Go straight, turn right, turn left |
| **Rewards** | +10 for food, -10 for death |
| **Algorithm** | DQN with experience replay, target network, epsilon-greedy exploration |

---

## Stack

- Python 3.11+
- PyTorch
- Pygame
- Matplotlib

---

## Links

- **Play locally:** [http://localhost:8080](http://localhost:8080) (after `python serve.py`)
- **Live site:** your `*.vercel.app` URL (static landing page)
- **GitHub:** [nilabh25bcs10235-beep/snake.rl](https://github.com/nilabh25bcs10235-beep/snake.rl)