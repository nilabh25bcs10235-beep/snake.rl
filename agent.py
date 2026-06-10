import copy
import random

import numpy as np
import torch
from collections import deque

from model import Linear_QNet, QTrainer

INPUT_SIZE = 11
HIDDEN_SIZE = 512
OUTPUT_SIZE = 3

MAX_MEMORY = 200_000
BATCH_SIZE = 2000
LR = 0.0003
GAMMA = 0.99


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.target_model = copy.deepcopy(self.model)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        self.trainer.sync_target()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            sample = list(self.memory)
        else:
            sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, eval_mode=False):
        if eval_mode:
            self.epsilon = 0
        else:
            self.epsilon = max(10, 80 - self.n_games)

        action = [0, 0, 0]

        if not eval_mode and random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state_t)
                idx = int(prediction.argmax().item())

        action[idx] = 1
        return action

    def load(self, path='./model/model.pth'):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.trainer.sync_target()