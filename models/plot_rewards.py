import matplotlib.pyplot as plt
import json


def plot_rewards(log_file="rewards_log.json", title="Episode Rewards", save_path=None):
    with open(log_file, "r") as f:
        rewards = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    plot_rewards()
