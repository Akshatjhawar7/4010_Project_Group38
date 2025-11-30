import gym
import numpy as np
import json
import os
from TrafficRLEnv import TrafficIntersectionEnv

# Fixed-time controller: switches every 15 steps
class FixedTimeAgent:
    def __init__(self, interval=15):
        self.interval = interval
        self.last_switch_step = 0

    def act(self, step_count):
        if (step_count - self.last_switch_step) >= self.interval:
            self.last_switch_step = step_count
            return 1  # switch
        return 0  # hold

# Setup
env = TrafficIntersectionEnv()
agent = FixedTimeAgent(interval=15)
EPISODES = 600
REWARD_LOG = []
all_metrics = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    steps = 200

    for step in range(200):
        action = agent.act(step)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            steps = step
            break

    # Metrics
    car_wait_avg = np.mean([sum(d['car_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    ped_wait_avg = np.mean([sum(d['ped_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    signal_switches = len(env.signal_switch_log)

    # Logging
    episode_metrics = {
        'episode': episode + 1,
        'reward': total_reward,
        'steps': steps,
        'car_wait_avg': car_wait_avg,
        'ped_wait_avg': ped_wait_avg,
        'signal_switches': signal_switches
    }
    all_metrics.append(episode_metrics)
    print(f"[Fixed] Episode {episode+1}, Reward: {total_reward:.2f}, Car Wait: {car_wait_avg:.2f}, Ped Wait: {ped_wait_avg:.2f}, Switches: {signal_switches}")

# Saving to JSON
os.makedirs("logs", exist_ok=True)
with open("logs/fixed_time_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
