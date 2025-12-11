# TrafficIntersectionEnv: RL-Based Traffic Signal Control with Pedestrians

This repository contains a custom OpenAI Gym environment for simulating a four-way traffic intersection with both vehicles and pedestrians, along with reinforcement learning agents (Q-Learning, SARSA, DQN) trained to optimize signal control for reduced delay and improved safety.

## Project Overview

- **Environment:** Custom Gym environment (`TrafficIntersectionEnv`) with support for vehicles and pedestrian queues.
- **Agents:** Classical tabular RL (Q-Learning, SARSA) and Deep Q-Network (DQN) with function approximation.
- **Goals:**
  - Reduce overall delay for both cars and pedestrians.
  - Learn realistic signal switching with yellow-phase safety constraints.
  - Compare RL against rule-based controllers.

---

---

## Installation

### Requirements

- Python
- pip
- pygame
- gym
- numpy
- torch

## Visualization
To visualize, simply run
python run_env.py

## Environment

The core environment with reward shaping is stored in TrafficRLEnv.py

## RL Agents

To train the RL agents, the files are-

- Q Learning: train_q_learning.py
- SARSA agent: sarsa_agent.py
- Deep Q Learning: dqn_agent.py

## Visualizing Metrics

For visualizing and graphing raw results, compare.py is used

## Evaluation Metrics

After training, each agent is evaluated based on:

- Total Episode Reward

- Average Vehicle Delay

- Average Pedestrian Delay

- Total Switches

Logs are saved in logs/ as .json files for further plotting or analysis.

## Results

RL agents significantly outperform rule-based policies in mixed traffic scenarios.

## Future Work
- Multi-agent intersection networks
- Multi-behavioural agents like ambulances, elderly pedestrians etc.



