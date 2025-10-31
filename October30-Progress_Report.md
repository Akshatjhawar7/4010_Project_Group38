# Traffic Signal Control Using Q-Learning – Progress Update (October 30)


**Team Members:**  
Akshat Jhawar

**Date:** October 30, 2025

---

## 1. Q-Learning Agent Implementation

- Replaced the earlier **random agent** with a **tabular Q-learning agent** capable of learning optimal signal switching policies.
- The agent interacts directly with the `TrafficIntersectionEnv` using **epsilon-greedy exploration**.
- Core parameters implemented:
  - Learning rate (`lr`): 0.1  
  - Discount factor (`gamma`): 0.99  
  - Epsilon decay (`decay`): 0.995  
  - Epsilon minimum (`epsilon_min`): 0.01  
- The Q-table updates dynamically as new state-action pairs are encountered.

---

## 2. State Representation Redesign

- Simplified and optimized the state structure for better generalization:
  - Combined **North+South** and **East+West** vehicle queues.
  - Grouped **pedestrian queues** for both directions.
  - Included **signal phase** and **timer buckets** (0–3s, 3–6s, >6s).
- This binning approach significantly reduced the state dimensionality, improving both **training stability** and **learning speed**.

---

## 3. Reward Logging & Metrics Tracking

- Implemented CSV-based logging (`q_learning_metrics.csv`) capturing:
  - `episode`, `reward`, `epsilon`, `car_wait_avg`, `ped_wait_avg`, `signal_switches`, `steps`
- Added dynamic visualization for training progress using Matplotlib.
- Enabled per-episode tracking of signal-switch frequency and agent behavior.

---

## 5. Performance Improvements

- **Reward evolution:**
  - Early episodes: ~ -1500 total reward.
  - Later episodes: ~ -100 total reward.
- **Epsilon Decay Trend:**  
  Demonstrates clear learning progression and confidence growth in the agent’s decisions.
- **Reduced Wait Times:**  
  Noticeable drop in both average car and pedestrian queue lengths as training progresses.

---

## Upcoming Tasks

1. Compare **Q-learning vs Fixed-time** metrics side-by-side:
   - Reward curves
   - Wait time trends
   - Signal switch frequency
2. Introduce **advanced RL models** (e.g., PPO or DQN) for further benchmarking. (If time permits)
3. Add **pedestrian crossing-specific penalties/rewards** for realism. (As part of reward experimentation)
5. Prepare a **final performance summary dashboard** and evaluation visuals.

---

## Files Added / Updated

- `TrafficRLEnv.py` → Environment updates for timing, logs, and safety control.  
- `train_q_learning.py` → Main Q-learning training loop with metrics.  
- `q_learning_metrics.csv` → Detailed per-episode statistics for analysis.
- `q_learning_curve.png` → Q-Learning curve over time
  

---
