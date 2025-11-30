# Traffic RL Project – Progress Report (Nov 30)

## Updates Completed

### 1. Q-Learning Enhancement
- Modified `train_q_learning.py` to:
  - Record training metrics during episodes.
  - Output metrics in structured `.json` format.

### 2. Logging Support for All Baselines
- Added consistent metric recording in:
  - `dqn.py` (Deep Q-Learning)
  - `fixed_time.py` (Fixed policy baseline)
  - `sarsa.py` (SARSA agent)
- All files now track and export:
  - **Average pedestrian wait time**
  - **Average car wait time**
  - **Number of signal switches**
  - **Average reward per episode**

### 3. Data Analysis and Visualization
- Parsed `.json` logs to:
  - Generate performance plots.
  - Compare agent behavior across baselines.
- charts include:
  - Pedestrian wait vs. episode
  - Car wait vs. episode
  - Signal switches vs. episode
  - Reward vs. episode

## Artifacts Generated
- `logs/`: Contains `.json` logs
- `metrics/`: Contains `.png` visualizations for comparisons

## Next Steps
- Make github readme with instructions to run
- Begin report writing with insights from the plots

---
