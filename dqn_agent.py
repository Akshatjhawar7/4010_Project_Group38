import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import json
from TrafficRLEnv import TrafficIntersectionEnv

# Defining Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.lr = 1e-3
        self.memory = deque(maxlen=10000)

        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training loop
def train_dqn(episodes=600):
    env = TrafficIntersectionEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    all_metrics = []

    for ep in range(episodes):
        s = env.reset()
        env.wait_time_log = []
        env.signal_switch_log = []
        total_reward = 0

        for _ in range(200):  # max steps
            a = agent.act(s)
            s_, r, done, _ = env.step(a)
            agent.remember(s, a, r, s_, done)
            agent.replay()
            s = s_
            total_reward += r

        # Storing metrics from env logs
        car_wait = np.mean([sum(x['car_waits']) for x in env.wait_time_log])
        ped_wait = np.mean([sum(x['ped_waits']) for x in env.wait_time_log])
        switch_count = len(env.signal_switch_log)

        all_metrics.append({
            "episode": ep + 1,
            "reward": total_reward,
            "car_wait_avg": car_wait,
            "ped_wait_avg": ped_wait,
            "signal_switches": switch_count
        })

        episode_rewards.append(total_reward)
        print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Car Wait: {car_wait:.2f} | Ped Wait: {ped_wait:.2f} | Switches: {switch_count}")

    # Saving metrics
    os.makedirs("logs", exist_ok=True)
    with open("logs/dqn_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

if __name__ == "__main__":
    train_dqn()
