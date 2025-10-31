import gym
import numpy as np
import matplotlib.pyplot as plt
from TrafficRLEnv import TrafficIntersectionEnv
import csv
import os

class QLearningAgent:
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

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)

# Training loop
env = TrafficIntersectionEnv()
agent = QLearningAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)
log_path = "q_learning_metrics.csv"
log_exists = os.path.isfile(log_path)

csv_file = open(log_path, mode='w', newline='')
fieldnames = ['episode', 'reward', 'epsilon', 'steps', 'car_wait_avg', 'ped_wait_avg', 'signal_switches']
log_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
log_writer.writeheader()

EPISODES = 500
REWARD_LOG = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    steps = 200

    for step in range(200):  # Max steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            steps = step
            break

    agent.update_epsilon()
    REWARD_LOG.append(total_reward)
    car_wait_avg = np.mean([sum(d['car_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    ped_wait_avg = np.mean([sum(d['ped_waits']) for d in env.wait_time_log]) if env.wait_time_log else 0
    signal_switches = len(env.signal_switch_log)

    log_writer.writerow({
        'episode': episode + 1,
        'reward': total_reward,
        'epsilon': agent.epsilon,
        'steps': steps,
        'car_wait_avg': car_wait_avg,
        'ped_wait_avg': ped_wait_avg,
        'signal_switches': signal_switches
    })

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

csv_file.close()

# Plotting rewards
plt.plot(REWARD_LOG)
plt.title("Q-Learning: Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
