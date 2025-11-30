import json
import matplotlib.pyplot as plt

# Paths to JSON logs
file_paths = {
    "Q-Learning": "q_learning_metrics.json",
    "DQN": "dqn_metrics.json",
    "Fixed-Time": "fixed_time_metrics.json",
    "SARSA": "sarsa_metrics.json"
}

# Loading data
metrics_data = {}
for agent, path in file_paths.items():
    with open(path, 'r') as f:
        metrics_data[agent] = json.load(f)

# Defining common settings
colors = {
    "Q-Learning": "tab:blue",
    "DQN": "tab:green",
    "Fixed-Time": "tab:orange",
    "SARSA": "tab:red"
}
metrics = {
    "reward": "Total Reward per Episode",
    "car_wait_avg": "Average Car Wait Time per Episode",
    "ped_wait_avg": "Average Pedestrian Wait Time per Episode",
    "signal_switches": "Average Signal Switches per Episodes"
}

# Creating a separate plot for each metric
for metric_key, title in metrics.items():
    plt.figure(figsize=(12, 6))
    for agent in metrics_data:
        data = metrics_data[agent]
        x = [ep["episode"] for ep in data]
        y = [ep[metric_key] for ep in data]
        plt.plot(x, y, label=agent, color=colors[agent])
    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel(metric_key.replace("_", " ").title(), fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
