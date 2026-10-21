# Finalised

# this code plots model scores qNIPV per each initial dataset size
# plots all kernels, qNIPV


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

# END OF PLOT SETTINGS ====

sampling_types = [
    "Sobol",
    "Manual"
    ]

initial_dataset_sizes = [
    25,
    50,
    100,
    200
    ]

data_plot_settings = {"25": ["#860ba4", "x"],
                      "50": ["#bfcf2c", "+"],
                     "100": ["#9d5d25", "^"],
                     "200": ["#15926f", "*"],}

for sampling_type in sampling_types:

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    axes = [ax1, ax2, ax3]
    metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
    metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

    for size in initial_dataset_sizes:
        data = pd.read_csv(f"active_learning_model/initial_dataset_size/{sampling_type}/{sampling_type}_initial_dataset_{size}_scores.csv")
        color_marker = data_plot_settings[f"{size}"]

        for i in range(len(axes)):
            ax = axes[i]
            metric = metrics[i]
            y_label = metric_labels[i]

            ax.plot(
                data["no_samples"], 
                data[metric],
                label=f"{size}",
                color=color_marker[0],
                marker=color_marker[1],
                markersize=5
            )
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(y_label)
            

    for i in range(len(axes)):
        ax = axes[i]
        metric = metrics[i]
        y_label = metric_labels[i]

        # Plot random samples
        random_scores = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_random.csv")
        rand_x = [50, 100, 200, 300] # no samples, no value for 25
        rand_y = [random_scores.loc[random_scores["no_samples"] == i, metric].values[0] for i in rand_x]

        ax.scatter(
            rand_x, 
            rand_y,
            label=f"Random",
            color="#d62728",
            marker="s")

        ax.legend(title="Initial dataset size", title_fontsize=14)

        # add annotations
        ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
        ax.set_box_aspect(1)

    fig.tight_layout()

    save_fig(fig, f"{sampling_type}_initial_dataset_plot", f"active_learning_model/initial_dataset_size/{sampling_type}")

