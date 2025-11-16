import gym
import numpy as np
import matplotlib.pyplot as plt
from TrafficRLEnv import TrafficIntersectionEnv
import csv
import os

# Fixed-time controller: switches every 15 steps
class FixedTimeAgent:
    def __init__(self, interval=15):
        self.interval = interval
        self.last_switch_step = 0
        self.current_action = 0  # 0: hold, 1: switch

    def act(self, step_count):
        # Switch every `interval` steps
        if (step_count - self.last_switch_step) >= self.interval:
            self.last_switch_step = step_count
            return 1  # switch
        return 0  # hold

# Setup
env = TrafficIntersectionEnv()
agent = FixedTimeAgent(interval=15)
log_path = "fixed_time_metrics.csv"
csv_file = open(log_path, mode='w', newline='')
fieldnames = ['episode', 'reward', 'steps', 'car_wait_avg', 'ped_wait_avg', 'signal_switches']
log_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
log_writer.writeheader()

EPISODES = 1000
REWARD_LOG = []

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

    REWARD_LOG.append(total_reward)
    car_wait_avg = np.mean([sum(d['car_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    ped_wait_avg = np.mean([sum(d['ped_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    signal_switches = len(env.signal_switch_log)

    log_writer.writerow({
        'episode': episode + 1,
        'reward': total_reward,
        'steps': steps,
        'car_wait_avg': car_wait_avg,
        'ped_wait_avg': ped_wait_avg,
        'signal_switches': signal_switches
    })

    if (episode + 1) % 50 == 0:
        print(f"[Fixed] Episode {episode+1}, Total Reward: {total_reward}")

csv_file.close()

# Plot reward trend
plt.plot(REWARD_LOG)
plt.title("Fixed-Time Agent (15s): Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
