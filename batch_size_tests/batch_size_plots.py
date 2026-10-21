# Finalised

# plotting metrics for each batch size as number of samples increases


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

# cycling through initial sizes
init_sizes = [
    50,
    100
    ]

for init_size in init_sizes:

    batch_sizes = [
        2,
        5,
        10,
        20,
        50
        ]

    data_plot_settings = {2: ["#a848c0", "x"],
                        5: ["#a1aa51", "+"],
                        10: ["#834a19", "^"],
                        20: ["#178163", "*"],
                        50: ["#4036AD", "."]}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    axes = [ax1, ax2, ax3]
    metrics = ["test_avg_r2", "test_avg_rmse", "scalar_mll"]
    metric_labels = [r"$\mathrm{R^2}$", "RMSE", "MLL"]

    for batch_size in batch_sizes:
        data = pd.read_csv(f"active_learning_model/batch_size_tests/Data{init_size}/batch_size_{batch_size}_scores.csv")
        color_marker = data_plot_settings[batch_size]

        for i in range(len(axes)):
            ax = axes[i]
            metric = metrics[i]
            y_label = metric_labels[i]

            ax.plot(
                data["no_samples"], 
                data[metric],
                label=f"{batch_size}",
                color=color_marker[0],
                marker=color_marker[1],
                markersize=5
            )
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel(y_label)
            ax.legend(title="Batch Size", title_fontsize=14)

            # add annotations
            ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
            ax.set_box_aspect(1)

    fig.tight_layout()

    save_fig(fig, f"batch_sizes_init{init_size}_plot", f"active_learning_model/batch_size_tests")