import matplotlib.pyplot as plt
import numpy as np


def visualize_regression_metrics(metrics, title):
    groups = list(metrics.keys())
    mse = [metrics[g]["MSE"] for g in groups]
    bias = [metrics[g]["Bias"] for g in groups]

    x = np.arange(len(groups))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, mse, width=0.4, label="MSE")
    plt.bar(x + 0.2, bias, width=0.4, label="Bias")

    plt.xticks(x, groups)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()

    return plt
