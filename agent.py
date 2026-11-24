import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from model import QNet
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, n_actions, obs_shape, device):
        self.n_actions = n_actions
        self.device = device

        self.q = QNet(n_actions).to(device)         # the brain being trained
        self.target = QNet(n_actions).to(device)    # slow-moving copy for stable targets
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.LEARNING_RATE)
        self.buffer = ReplayBuffer(config.BUFFER_SIZE, obs_shape)

    def epsilon(self, step, decay_steps):
        """Chance of a random move: starts high, falls linearly to a small value."""
        frac = min(1.0, step / decay_steps)
        return config.EPS_START + frac * (config.EPS_END - config.EPS_START)

    def act(self, state, eps):
        if random.random() < eps:                       # explore
            return random.randrange(self.n_actions)
        with torch.no_grad():                           # exploit
            s = torch.as_tensor(np.asarray(state), device=self.device).unsqueeze(0)
            return self.q(s).argmax(dim=1).item()

    def learn(self):
        s, a, r, s2, d = self.buffer.sample(config.BATCH_SIZE)
        s  = torch.as_tensor(s,  device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        a  = torch.as_tensor(a,  device=self.device)
        r  = torch.as_tensor(r,  device=self.device)
        d  = torch.as_tensor(d,  device=self.device)

        # What the network currently predicts for the actions it actually took
        predicted = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # A better estimate: reward now + discounted value of the best next move.
        # Double DQN: the main network PICKS the next action, the target network SCORES it.
        with torch.no_grad():
            best_next = self.q(s2).argmax(dim=1, keepdim=True)
            next_value = self.target(s2).gather(1, best_next).squeeze(1)
            wanted = r + config.GAMMA * next_value * (1.0 - d)

        loss = F.smooth_l1_loss(predicted, wanted)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()