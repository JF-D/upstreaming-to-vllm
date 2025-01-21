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


def plot_subgraph(ax: plt.Axes, all_ys, data_tags, title):
    width = 0.3

    for i, dt in enumerate(data_tags):
        xpos = [j for j in range(len(all_ys[i]))]
        ys = all_ys[i]
        if i > 0:
            botty = all_ys[i - 1]
            for j in range(i - 1):
                botty = [a + b for a, b in zip(botty, all_ys[j])]
        else:
            botty = None
        ax.bar(xpos, ys, width, bottom=botty, label=dt, color=color_palette[i])

    ax.set_xticks([i for i in range(len(all_ys[0]))])


def plot_full_prefill_breakdown():
    # full prefill latency
    p4d = {
        "total": 135.7,
        "Comm": 39.28,
        "Attn": 0.547 * 32,
        "MLP": 1.7068 * 32,
    }
    p4d["Other"] = p4d["total"] - p4d["Comm"] - p4d["Attn"] - p4d["MLP"]

    trn1_mlp8 = {
        "total": 209.77,
        "Comm": 64.08,
        "Attn": 2.16*32,
        "MLP": 1.37 * 32,
    }
    trn1_mlp8["Other"] = trn1_mlp8["total"] - trn1_mlp8["Comm"] - trn1_mlp8["Attn"] - trn1_mlp8["MLP"]

    trn1_mlp1 = {
        "total": 294.6,
        "Comm": 116.08,
        "Attn": 2.12*32,
        "MLP": 1.95 * 32,
    }
    trn1_mlp1["Other"] = trn1_mlp1["total"] - trn1_mlp1["Comm"] - trn1_mlp1["Attn"] - trn1_mlp1["MLP"]

    data_tags = ["MLP", "Attn", "Comm", "Other"]
    full_prefill_p4d = [p4d[dt] for dt in data_tags]
    full_prefill_trn1_mlp1 = [trn1_mlp1[dt] for dt in data_tags]
    full_prefill_trn1_mlp8 = [trn1_mlp8[dt] for dt in data_tags]

    # append prefill latency
    p4d = {
        "total": 84,
        "Comm": 14.686,
        "Attn": 1.267 * 32,
        "MLP": 0.527 * 32,
    }
    p4d["Other"] = p4d["total"] - p4d["Comm"] - p4d["Attn"] - p4d["MLP"]

    trn1_mlp4 = {
        "total": 112.94,
        "Comm": 26.76,
        "Attn": 1.97 * 32,
        "MLP": 0.458 * 32,
    }
    trn1_mlp4["Other"] = trn1_mlp4["total"] - trn1_mlp4["Comm"] - trn1_mlp4["Attn"] - trn1_mlp4["MLP"]

    trn1_mlp1 = {
        "total": 138.3,
        "Comm": 49.6,
        "Attn": 1.96 * 32,
        "MLP": 0.445 * 32,
    }
    trn1_mlp1["Other"] = trn1_mlp1["total"] - trn1_mlp1["Comm"] - trn1_mlp1["Attn"] - trn1_mlp1["MLP"]
    append_prefill_p4d = [p4d[dt] for dt in data_tags]
    append_prefill_trn1_mlp1 = [trn1_mlp1[dt] for dt in data_tags]
    append_prefill_trn1_mlp4 = [trn1_mlp4[dt] for dt in data_tags]

    fig, axs = plt.subplots(1, 2, dpi=300)
    fig.set_size_inches(6, 3)
    ax = axs[0]
    for i, tag in enumerate(data_tags):
        all_ys = []
        for j in range(len(data_tags)):
            y_data = [full_prefill_p4d[j], full_prefill_trn1_mlp8[j], full_prefill_trn1_mlp1[j]]
            all_ys.append(y_data)

        plot_subgraph(ax, all_ys, data_tags, "")
    ax.set_xticklabels([f'vLLM (p4d)', 'Our (SP8)', 'Our (TP)'], fontsize=10)
    ax.set_ylabel("Latency (ms)")
    
    ax = axs[1]
    for i, tag in enumerate(data_tags):
        all_ys = []
        for j in range(len(data_tags)):
            y_data = [append_prefill_p4d[j], append_prefill_trn1_mlp4[j], append_prefill_trn1_mlp1[j]]
            all_ys.append(y_data)

        plot_subgraph(ax, all_ys, data_tags, "")
    ax.set_xticklabels([f'vLLM (p4d)', 'Our (SP4)', 'Our (TP)'], fontsize=10)
    
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[:4], labels[:4]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncols=4)

    fig.savefig("experiments/graphs/latency-breakdown-8B.pdf")


if __name__ == "__main__":
    plot_full_prefill_breakdown()