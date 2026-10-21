# Finalised

# plots model metric comparisons of different kernels

import os
import matplotlib.pyplot as plt
import string
import numpy as np
import pandas as pd

# Defining plot settings ======================================================================================

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

# function to plot pred vs actual for each kernel ---------------------------------------------------
def plot_kernel_folds(name, plot_title, pred, actual, type):

    if type == "RBF":
         color="#1f77b4"
    elif type == "Matern_32":
         color="#ff7f0e"
    elif type == "Matern_52":
         color="#2ca02c"
    else:
         print("Invalid covar type.")
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(
        pred, actual,
        marker=".",
        s=80,
        color=color,
        alpha=0.7
    )
        
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5)
    ax.set_title(f"{plot_title}")
    ax.set_xlabel("Actual P2/O3")
    ax.set_ylabel("Predicted P2/O3")
    # ax.legend(loc="upper left")

    save_fig(fig, name=f"{name}_plot_v3", folder="surrogate_design/kernel_testing/kernel_plots")

# DATA TO PLOT =========================================================================

data = pd.read_csv("active_learning_model/surrogate_design/kernel_testing/surrogate_test_results_v3.csv")

# Only take scaled ARD kernels with low noise for RBF, Matern 32, Matern 52

kernels = ["RBF",
           "Scaled RBF ARD",
           "Matern 32",
           "Scaled Matern 32 ARD",
           "Matern 52",
           "Scaled Matern 52 ARD"]

colors = ["#1f77b4", "#1f77b4",
          "#ff7f0e", "#ff7f0e",
          "#2ca02c", "#2ca02c"]

markers = [".", "o",
           "d", "D",
           "*", "^"]

s=120

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))

# annotations
axes = [ax1, ax2]
for i in range(2):
    ax = axes[i]

    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)


for i in range(len(kernels)):

    kernel = kernels[i]
    color = colors[i]
    marker = markers[i]

    ax1.scatter(data.loc[data["kernel_name"] == kernel + " Low Noise", "scalar_mll"], 
                data.loc[data["kernel_name"] == kernel + " Low Noise", "test_avg_r2"],
                color=color, marker=marker, label=kernel, s=s, alpha=0.7)
    ax1.set_ylim(0.45,0.65)
    ax1.set_xlabel("Scalar MLL")
    ax1.set_ylabel("$R^2$")
    ax1.set_box_aspect(1)

    ax2.scatter(data.loc[data["kernel_name"] == kernel + " Low Noise", "scalar_mll"],
                data.loc[data["kernel_name"] == kernel + " Low Noise", "test_avg_rmse"],
                color=color, marker=marker, label=kernel, s=s, alpha=0.7)
    ax2.set_ylim(0.22,0.25)
    ax2.set_xlabel("Scalar MLL")
    ax2.set_ylabel("RMSE")
    ax2.set_box_aspect(1)

    # adding a dotted line between the same kernel types
    x1 = data.loc[data["kernel_name"] == kernel + " Low Noise", "scalar_mll"]
    y1 = data.loc[data["kernel_name"] == kernel + " Low Noise", "test_avg_r2"]

    x2 = data.loc[data["kernel_name"] == kernel + " Low Noise", "scalar_mll"]
    y2 = data.loc[data["kernel_name"] == kernel + " Low Noise", "test_avg_rmse"]

    if i % 2 == 0 and (i + 1) < len(kernels):
        next_kernel = kernels[i + 1]
        
        # Get coordinates for paired point (i+1)
        next_x1 = data.loc[data["kernel_name"] == next_kernel + " Low Noise", "scalar_mll"]
        next_y1 = data.loc[data["kernel_name"] == next_kernel + " Low Noise", "test_avg_r2"]
        
        next_x2 = data.loc[data["kernel_name"] == next_kernel + " Low Noise", "scalar_mll"]
        next_y2 = data.loc[data["kernel_name"] == next_kernel + " Low Noise", "test_avg_rmse"]
        
        # connect current point (i) to the paired point (i+1)
        ax1.plot([x1.values[0], next_x1.values[0]], [y1.values[0], next_y1.values[0]], 
                 color=color, linestyle="--", alpha=0.3, zorder=1)
                 
        ax2.plot([x2.values[0], next_x2.values[0]], [y2.values[0], next_y2.values[0]], 
                 color=color, linestyle="--", alpha=0.3, zorder=1)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', 
           ncol=len(kernels)/2, fontsize=14)
plt.tight_layout()

save_fig(fig, name="kernel_results_plot", folder="surrogate_design/kernel_testing")