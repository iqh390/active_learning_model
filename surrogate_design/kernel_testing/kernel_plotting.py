# Finalised

# Plot figures for all kernels (pred v actual)

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string


# Defining plot settings ======================================================================================

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

def plot_settings():

    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
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
def plot_kernel_folds(name, plot_title, pred, actual, r_sq, rmse, type, save=True):

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
    ax.text(0.95, 0.05, f"$R^2 = {r_sq:.3f}$\n$\mathrm{{RMSE}} = {rmse:.3f}$", 
            transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=14)
    if save == True:
        save_fig(fig, name=f"{name}_plot_v3", folder="active_learning_model/surrogate_design/kernel_testing/kernel_plots")
    else:
         return fig, ax

# DATA AND PLOTTING========================================================================

all_kernel_settings = [
    ("RBF", {"type": "RBF", "scaled": False, "ard": False}),
    ("RBF ARD", {"type": "RBF", "scaled": False, "ard": True}),
    ("Scaled RBF", {"type": "RBF", "scaled": True, "ard": False}),
    ("Scaled RBF ARD", {"type": "RBF", "scaled": True, "ard": True}),

    ("Matern 32", {"type": "Matern_32", "scaled": False, "ard": False}),
    ("Matern 32 ARD", {"type": "Matern_32", "scaled": False, "ard": True}),
    ("Scaled Matern 32", {"type": "Matern_32", "scaled": True, "ard": False}),
    ("Scaled Matern 32 ARD", {"type": "Matern_32", "scaled": True, "ard": True}),

    ("Matern 52", {"type": "Matern_52", "scaled": False, "ard": False}),
    ("Matern 52 ARD", {"type": "Matern_52", "scaled": False, "ard": True}),
    ("Scaled Matern 52", {"type": "Matern_52", "scaled": True, "ard": False}),
    ("Scaled Matern 52 ARD", {"type": "Matern_52", "scaled": True, "ard": True})
    ]


for name, kernel_settings in all_kernel_settings:

        # get metrics
        metrics = pd.read_csv("active_learning_model/surrogate_design/kernel_testing/surrogate_test_results_v3.csv")

        name1 = f"{name} Low Noise"

        white_noise_threshold = 1E-04
        data = pd.read_csv(f"active_learning_model/surrogate_design/kernel_testing/kernel_plot_data/{name} Low Noise_data.csv")

        # get model metrics
        r_sq1 = float(metrics.loc[metrics["kernel_name"] == name1, "test_avg_r2"].values[0])
        rmse1 = float(metrics.loc[metrics["kernel_name"] == name1, "test_avg_rmse"].values[0])

        # Plot using function
        plot_kernel_folds(name1, name, data["act_P2O3"], data["pred_P2O3"],
                          r_sq1, rmse1,
                          type=kernel_settings["type"],
                          save=True)

        name2 = f"{name} High Noise"
        # Change white noise threshold
        white_noise_threshold = 0.0001
        data_more_noise = pd.read_csv(f"active_learning_model/surrogate_design/kernel_testing/kernel_plot_data/{name} High Noise_data.csv")

        # get model metrics
        r_sq2 = float(metrics.loc[metrics["kernel_name"] == name2, "test_avg_r2"].values[0])
        rmse2 = float(metrics.loc[metrics["kernel_name"] == name2, "test_avg_rmse"].values[0])

        plot_kernel_folds(name2, name, data["act_P2O3"], data["pred_P2O3"],
                          r_sq2, rmse2,
                          type=kernel_settings["type"],
                          save=True)


# PLOT COMPARING RBF, MATERN52, MATERN32, w/ and w/o scaling and ARD ==================

select_kernels = [
    ("RBF", {"type": "RBF", "scaled": False, "ard": False}),
    ("Matern 32", {"type": "Matern_32", "scaled": False, "ard": False}),
    ("Matern 52", {"type": "Matern_52", "scaled": False, "ard": False}),
    ("Scaled RBF ARD", {"type": "RBF", "scaled": True, "ard": True}),
    ("Scaled Matern 32 ARD", {"type": "Matern_32", "scaled": True, "ard": True}),
    ("Scaled Matern 52 ARD", {"type": "Matern_52", "scaled": True, "ard": True})
]

fig2, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 14))

axes = [ax1, ax2, ax3, ax4, ax5, ax6]
i = 0

for name, kernel_settings in select_kernels:
            
        ax = axes[i]

        # get metric data
        metrics = pd.read_csv("active_learning_model/surrogate_design/kernel_testing/surrogate_test_results_v3.csv")

        name1 = f"{name} Low Noise"

        white_noise_threshold = 1E-04
        data = pd.read_csv(f"active_learning_model/surrogate_design/kernel_testing/kernel_plot_data/{name} Low Noise_data.csv")

        # get model metrics
        r_sq1 = float(metrics.loc[metrics["kernel_name"] == name1, "test_avg_r2"].values[0])
        rmse1 = float(metrics.loc[metrics["kernel_name"] == name1, "test_avg_rmse"].values[0])

        # Plot using function
        new_ax = plot_kernel_folds(name1, name, data["act_P2O3"], data["pred_P2O3"],
                        r_sq1, rmse1,
                        type=kernel_settings["type"],
                        save=False)
        
        if kernel_settings["type"] == "RBF":
            color="#1f77b4"
        elif kernel_settings["type"] == "Matern_32":
            color="#ff7f0e"
        elif kernel_settings["type"] == "Matern_52":
            color="#2ca02c"
        else:
            print("Invalid covar type.")
        
        ax.scatter(
            data["act_P2O3"], data["pred_P2O3"],
            marker=".",
            s=80,
            color=color,
            alpha=0.7
        )
            
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5)
        ax.set_title(f"{name}")
        ax.set_xlabel("Actual P2/O3")
        ax.set_ylabel("Predicted P2/O3")
        ax.text(0.95, 0.05, f"$R^2 = {r_sq1:.3f}$\n$\mathrm{{RMSE}} = {rmse1:.3f}$", 
                transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=14)
        i += 1

for i in range(len(axes)):
    ax = axes[i]
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

save_fig(fig2, name="full_kernel_plots", folder="active_learning_model/surrogate_design/kernel_testing/kernel_plots")
            