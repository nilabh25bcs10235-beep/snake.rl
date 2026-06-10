import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, filename='model.pth'):
        folder = './model'
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, filename))


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.learn_steps = 0

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1)
            next_q = self.target_model(next_state).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * next_q[i]
            target[i][action[i].argmax().item()] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % 500 == 0:
            self.sync_target()

        return float(loss.item())