import matplotlib.pyplot as plt
import json

def load_rewards(filename):
    with open(filename, "r") as f:
        return json.load(f)

def smooth(y, weight=0.9):
    smoothed = []
    last = y[0]
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_comparison():
    nd = load_rewards("rewards_log.json")
    linear = load_rewards("../examples/rewards_log_linear.json")

    plt.figure(figsize=(12, 6))
    plt.plot(smooth(nd), label="NdLinear", linewidth=2)
    plt.plot(smooth(linear), label="nn.Linear", linewidth=2)
    plt.title("NdLinear vs nn.Linear Performance on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison()
