# Finalised

# plots all kernels, qNIPV

import os
import matplotlib.pyplot as plt
import string
import pandas as pd
import numpy as np

# PLOT SETTINGS ====

# consistent colour palette
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

def plot_settings():
    plt.rcParams.update({
        "font.family": "Arial",
        "figure.figsize": (6, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
    })

def save_fig(fig, name, folder=None):
    if folder is not None:
        folder = f"{folder}/"

    output_path = os.path.join("", f"{folder}{name}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved {name}.")

plot_settings()

# PLOTTING ==============

# 1. Plotting all sampling type scores (R2, RMSE, MLL) for each acquisition function, comparing kernels

acqfs = ["qNIPV",
         "qPSD",
         "qUCB"]

for acqf in acqfs:

    rbf_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/RBF/synth_dataset_{acqf}_scores.csv")
    Matern52_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/Matern52/synth_dataset_{acqf}_scores.csv")
    Matern32_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/Matern32/synth_dataset_{acqf}_scores.csv")

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))

    axes = [ax1, ax2, ax3]
    metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
    metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

    labels = ["RBF", "Matern52", "Matern32"]
    scores = {"RBF": rbf_scores,
            "Matern52": Matern52_scores,
            "Matern32": Matern32_scores}

    data_plot_settings = {"RBF": ["#1f77b4", "x"],
                        "Matern52": ["#2ca02c", "+"],
                        "Matern32": ["#ff7f0e", "^"]}

    for label in labels:
        data = scores[label]
        color_marker = data_plot_settings[label]

        for j in range(len(metrics)):
            metric = metrics[j]
            y_label = metric_labels[j]
            ax = axes[j]

            ax.plot(
                data["no_samples"], 
                data[metric],
                label=label,
                color=color_marker[0],
                marker=color_marker[1],
                markersize=5
            )
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(y_label)
            ax.legend()

            ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[j] + ")", transform=ax.transAxes, size=16, weight='bold')
            ax.set_box_aspect(1)

    fig1.tight_layout()

    save_fig(fig1, f"{acqf}_kernel_comparison_plot", "active_learning_model/active_learning_model_synth")

# 2. Plotting for each kernel to compare acquisition functions


kernels = ["RBF",
           "Matern52",
           "Matern32"
           ]

for kernel in kernels:

    qNIPV_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qNIPV_scores.csv")
    qPSD_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qPSD_scores.csv")
    qUCB_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qUCB_scores.csv")

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    axes = [ax1, ax2, ax3]
    metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
    metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

    labels = ["qNIPV",
              "qPSD",
              "qUCB"]
    
    scores = {"qNIPV": qNIPV_scores,
            "qPSD": qPSD_scores,
            "qUCB": qUCB_scores}

    data_plot_settings = {"qNIPV": ["#008000", "d"],
                        "qPSD": ["#800080", "o"],
                        "qUCB": ["#FFA500", "s"]}

    for label in labels:
        data = scores[label]
        color_marker = data_plot_settings[label]

        for j in range(len(metrics)):
            metric = metrics[j]
            y_label = metric_labels[j]
            ax = axes[j]

            ax.plot(
                data["no_samples"], 
                data[metric],
                label=label,
                color=color_marker[0],
                marker=color_marker[1],
                markersize=4
            )
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(y_label)
            ax.legend()

            # add annotations and make square
            ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[j] + ")", transform=ax.transAxes, size=16, weight='bold')
            ax.set_box_aspect(1)

    fig2.tight_layout()

    save_fig(fig2, f"{kernel}_acqf_comparison_plot", "active_learning_model/active_learning_model_synth")

# 3. Plotting full plot for acquisition functions, per kernel

kernels = ["RBF",
           "Matern52",
           "Matern32"
           ]

fig3, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 22))

all_axes = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]]

for i in range(len(kernels)):

    kernel = kernels[i]
    axes = all_axes[i]

    qNIPV_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qNIPV_scores.csv")
    qPSD_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qPSD_scores.csv")
    qUCB_scores = pd.read_csv(f"active_learning_model/active_learning_model_synth/{kernel}/synth_dataset_qUCB_scores.csv")

    metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
    metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

    labels = ["qNIPV",
              "qPSD",
              "qUCB"]
    
    scores = {"qNIPV": qNIPV_scores,
            "qPSD": qPSD_scores,
            "qUCB": qUCB_scores}

    data_plot_settings = {"qNIPV": ["#008000", "d"],
                        "qPSD": ["#800080", "o"],
                        "qUCB": ["#FFA500", "s"]}

    for label in labels:
        data = scores[label]
        color_marker = data_plot_settings[label]

        for j in range(len(metrics)):
            metric = metrics[j]
            y_label = metric_labels[j]
            ax = axes[j]

            ax.plot(
                data["no_samples"], 
                data[metric],
                label=label,
                color=color_marker[0],
                marker=color_marker[1],
                markersize=4
            )
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(y_label)
            ax.set_title(f"{kernel}")
            ax.legend()

    fig2.tight_layout()


flat_axes = np.array(all_axes).flatten()

for i in range(len(flat_axes)):
            ax = flat_axes[i]
            ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
            ax.set_box_aspect(1)

save_fig(fig3, f"all_kernels_acqf_comparison_plot", "active_learning_model/active_learning_model_synth")
