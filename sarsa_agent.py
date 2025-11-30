import gym
import numpy as np
import json
import os
from TrafficRLEnv import TrafficIntersectionEnv

class SARSAAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.q_table = {}

    def get_state_key(self, state):
        car_queues = state[:4]
        ped_queues = state[4:8]
        phase = state[16]
        timer = state[17]

        def bin_sum(arr, bins=[0, 1, 4, 7]):
            total = np.sum(arr)
            for i, b in enumerate(bins):
                if total <= b:
                    return i
            return len(bins)

        return (
            bin_sum(car_queues[[0, 2]]),  # N+S
            bin_sum(car_queues[[1, 3]]),  # E+W
            bin_sum(ped_queues[[0, 2]]),
            bin_sum(ped_queues[[1, 3]]),
            int(phase),
            0 if timer < 3 else (1 if timer < 6 else 2)
        )

    def act(self, state):
        state_key = self.get_state_key(state)
        if np.random.rand() < self.epsilon or state_key not in self.q_table:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, next_action):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        target = reward + self.gamma * self.q_table[next_key][next_action]
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

# Environment setup
env = TrafficIntersectionEnv()
agent = SARSAAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

EPISODES = 600
all_metrics = []

for episode in range(1, EPISODES + 1):
    state = env.reset()
    total_reward = 0
    env.wait_time_log = []
    env.signal_switch_log = []

    action = agent.act(state)

    for step in range(200):
        next_state, reward, done, _ = env.step(action)
        next_action = agent.act(next_state)

        agent.learn(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action
        total_reward += reward

        if done:
            break

    agent.update_epsilon()

    car_wait_avg = np.mean([sum(d['car_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    ped_wait_avg = np.mean([sum(d['ped_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    signal_switches = len(env.signal_switch_log)

    episode_metrics = {
        "episode": episode,
        "reward": total_reward,
        "epsilon": round(agent.epsilon, 4),
        "steps": step + 1,
        "car_wait_avg": car_wait_avg,
        "ped_wait_avg": ped_wait_avg,
        "signal_switches": signal_switches
    }

    all_metrics.append(episode_metrics)
    print(f"[SARSA] Episode {episode}, Reward: {total_reward:.2f}, "
              f"Car Wait: {car_wait_avg:.2f}, Ped Wait: {ped_wait_avg:.2f}, Switches: {signal_switches}")

# Saving JSON
os.makedirs("logs", exist_ok=True)
with open("logs/sarsa_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
