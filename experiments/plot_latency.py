import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# plt.rcParams["font.family"] = "Calibri"

color_palette = seaborn.color_palette('Paired', 25).as_hex()
# color_palette[2] = color_palette[3]
# color_palette = seaborn.color_palette().as_hex()
hatchs = ['////', '', '\\\\\\\\', '', 'o', 'O', '.', '*']
mymarkers = ['*', 'o', '+', 'x']


def plot_fig(fig, ax: plt.Axes, x_ticks, x_label, ys, y_label, speedups=None):
    width = 0.2
    x = np.arange(len(x_ticks)) - width * (len(ys) - 1) / 2
    for i, (label, y) in enumerate(ys.items()):
        ax.bar(x, y, width, label=label, color=color_palette[i])
        if speedups is not None:
            speedup = speedups[label]
            for j, sp in enumerate(speedup):
                if y[j] == 0: continue
                ax.text(x[j], y[j] + 5, f'{sp:.2f}x', color='black', ha='center', rotation=90, fontsize=8)
        x += width
    ax.set_xticks(np.arange(len(x_ticks)), x_ticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    

def plot_full_prefill_8B():
    # sequence length
    x = [1024, 2048, 4096, 8192, 16384, 32768]
    x_ticks = ["1K", "2K", "4K", "8K", "16K", "32K"]
    x_label = "Seq Len"
    data = {
        "TNx": [81.3, 101.5, 183.86, 407.46, 0, 0],
        "TNx w/ nki": [57.55, 98.75, 172.59, 339.17, 678.14, 0],
        "our": [36.59, 56.45, 107.32, 209.3, 425.96, 857.11],
        "vLLM (p4d)": [30.37, 40.06, 69.89, 131.04, 277.54, 633.24],
    }
    y_label = "Latency (ms)"
    baseline = data["TNx"].copy()
    baseline[-2] = data["TNx w/ nki"][-2]
    baseline[-1] = data["our"][-1]
    speedups = {}
    for key, value in data.items():
        speedups[key] = [v2 / v1 for v1, v2 in zip(value, baseline) if v1 != 0]

    fig, ax = plt.subplots(1, 1, dpi=300)
    plot_fig(fig, ax, x_ticks, x_label, data, y_label, speedups)
    fig.savefig("experiments/graphs/full_prefill_latency-8B.pdf")


def plot_append_prefill_8B():
    # sequence length
    x = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    x_ticks = ["4K", "8K", "16K", "32K", "64K", "128K", "256K"]
    x_label = "Seq Len"
    data = {
        "TNx": [83.93, 101.42, 157.96, 246.57, 511, 1028.18, 0],
        "TNx w/ nki": [85.79, 98.49, 124.28, 183.03, 358.76, 644.2, 0],
        "our": [70.15, 70.6, 85.33, 113.15, 168.2, 280.1, 503.16],
        "vLLM (p4d)": [46.4, 50.02, 59.34, 78.37, 115.87, 191.82, 345.05],
    }
    y_label = "Latency (ms)"
    baseline = data["TNx"].copy()
    baseline[-1] = data["our"][-1]
    speedups = {}
    for key, value in data.items():
        speedups[key] = [v2 / v1 for v1, v2 in zip(value, baseline) if v1 != 0]

    fig, ax = plt.subplots(1, 1, dpi=300)
    plot_fig(fig, ax, x_ticks, x_label, data, y_label, speedups)
    fig.savefig("experiments/graphs/append_prefill_latency-8B.pdf")


if __name__ == "__main__":
    # plot_full_prefill_8B()
    plot_append_prefill_8B()