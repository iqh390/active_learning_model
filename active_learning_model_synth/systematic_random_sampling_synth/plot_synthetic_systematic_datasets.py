# Finalised

# Plotting synthetic and random dataset scores

import os
import matplotlib.pyplot as plt
import string
import pandas as pd

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

# LOADING DATASETS ==========================================

exp_scores = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_exp.csv")
lhs_scores = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_lhs.csv")
sobol_scores = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_sobol2.csv")
rand_scores = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_random.csv")
rbf_qnipv_scores = pd.read_csv("active_learning_model/active_learning_model_synth/RBF/synth_dataset_qNIPV_scores.csv")

# 1. Plotting all sampling type scores (R2, RMSE, MLL)

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

axes = [ax1, ax2, ax3]
metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

labels = ["RBF+qNIPV", "Manual", "LHS", "Sobol", "Random"]
scores = {
    "RBF+qNIPV": rbf_qnipv_scores,
    "Manual": exp_scores,
    "LHS": lhs_scores,
    "Sobol": sobol_scores,
    "Random": rand_scores,
    }

data_plot_settings = {"RBF+qNIPV": ["#008000", "d"],
                      "Manual": ["#7f7f7f", "x"],
                      "LHS": ["#e377c2", "+"],
                      "Sobol": ["#1f77b4", "^"],
                      "Random": ["#d62728", "*"]}

for label in labels:
    if label == "RBF+qNIPV":
        linestyle = ":"
        markersize=2
    else:
        linestyle = "-"
        markersize=5

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
            markersize=markersize,
            linestyle=linestyle
        )
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(y_label)
        ax.legend()

for i in range(len(axes)):
    # add annotations
    ax = axes[i]
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

fig1.tight_layout()

save_fig(fig1, "systematic_random_sampling_synthetic_comparison_plot2", "active_learning_model/active_learning_model_synth/systematic_random_sampling_synth")
    
