# Finalised

import os
import matplotlib.pyplot as plt
import pandas as pd
import string
import numpy as np

# PLOT SETTINGS ====

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

# LOAD DATA ================================================

# 1. Plotting actual P2/O3 vs synthetic P2/O3 for the manual sampling dataset

fig1, ax = plt.subplots(figsize = (8, 6))

actual_exp_data = pd.read_csv("active_learning_model/original_dataset/df_exp_semi_refined.csv")
simulated_exp_data = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_expr.csv")

# FIG 1: Bar chart of P2O3 ratio for actual and RF synthetic data, and GP synthetic data --------------------------------
# note this uses RF simluated data (classification + regression) and includes incorrect Cat_Success predictions

# calculate the proportion of P2, O3 and mixed values in each bin
# input is a dataframe
def p2o3ratio_props(P2O3ratio, binwidth: int=0.1):

    unsuccessful = np.sum(P2O3ratio == -1)
    o3 = np.sum(P2O3ratio == 0)
    mixed = P2O3ratio[(P2O3ratio > 0) & (P2O3ratio < 1)].to_numpy()
    p2 = np.sum(P2O3ratio == 1)

    # counting each bin for mixed values
    bins = np.arange(0, 1.01, binwidth)
    counts, _ = np.histogram(mixed, bins=bins)

    return np.concatenate(([o3], counts, [p2], [unsuccessful]))

exp_P2O3_prop = p2o3ratio_props(actual_exp_data["P2O3ratio"])
print("Experimental")
print(exp_P2O3_prop)

synth_P2O3_prop = p2o3ratio_props(simulated_exp_data["P2O3ratio"])
print("Synthetic")
print(synth_P2O3_prop)

bar_width = 0.4
x = np.arange(13)

fig1, ax1 = plt.subplots(figsize=(10, 6))

# function to lighten colours for hatching
import matplotlib.colors as mc
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.
    amount=0 (original color), amount=1 (white)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)

ax1.bar(x - bar_width/2, exp_P2O3_prop, width=bar_width, color="#8e5ac5", hatch="\\", edgecolor=lighten_color("#8e5ac5", 0.3), label="Experimental")
ax1.bar(x + bar_width/2, synth_P2O3_prop, width=bar_width, color="#402959", hatch="xx", edgecolor=lighten_color("#402959", 0.3), label="Synthetic")

ax1.set_xticks(x)
ax1.set_xticklabels(
    ["0"] +
    [f"({i/10:.1f}, {(i+1)/10:.1f}]" for i in range(9)] +
    [f"({9./10:.1f}, {(9.+1)/10:.1f})"] +
    ["1"] +
    ["Unsuccessful"],
    rotation=45,
    ha="right"
)

ax1.set_xlabel("P2/O3")
ax1.set_ylabel("Frequency")
ax1.legend()

ax1.set_frame_on(True)
plt.tight_layout()

save_fig(fig1, "bar_chart_synth_vs_expr_P2O3_values", "active_learning_model/active_learning_model_expr")

# FIG 2: pred vs actual scatter plots for actual expr vs RF synthetic -------------------------------------

fig2, ax2 = plt.subplots(figsize=(6, 6))

ax2.scatter(simulated_exp_data["P2O3ratio"], actual_exp_data["P2O3ratio"], alpha=0.7, color="#8e5ac5", marker=".", s=80)
ax2.plot([-1, 1], [-1, 1], color="grey", linestyle="--", alpha=0.5)

ax2.set_xlabel('Predicted P2/O3')
ax2.set_ylabel('Actual P2/O3')
ax2.set_box_aspect(1)
plt.tight_layout()

save_fig(fig2, "scatter_expr_vs_synth_P2O3_values", "active_learning_model/active_learning_model_expr")

# FIG 3: Plotting GP predicted P2/O3 for each acquisition function -------------------------------
# only plots run 1

# function to set tick labels based on choice of binwidth
def p2o3_xticklabels(x, binwidth):
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["0"] +
        [f"({i/(1/binwidth):.1f}, {(i+1)/(1/binwidth):.1f}]" for i in range(int(1/binwidth) - 1)] +
        [f"({1 - binwidth:.1f}, 1.0]"] +
        ["1"] +
        ["Unsuccessful"],
        rotation=45,
        ha="right"
    )
    return

kernels = [
    "RBF", 
    "Matern52", 
    "Matern32"
    ]

acqfs = [
    "qNIPV", 
    "qUCB", 
    "qPSD"
    ]

fig3, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 26))
axes = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]]

colors = ["#8e5ac5", "#3F93A0", "#402959"]
hatches = ["\\", "xx", "++"]

for run in range(0, 3):
    color = colors[run]
    hatch = hatches[run]

    for i in range(len(kernels)):
        kernel = kernels[i]

        for j in range(len(acqfs)):
            ax = axes[i][j]
            acqf = acqfs[j]
            binwidth = 0.2
            bar_width=0.2

            data = pd.read_csv(f"active_learning_model/active_learning_model_expr/{kernel}/{kernel}_whole_dataset_{acqf}_bulk_candidates_run{run+1}.csv")
            p2o3ratio_plot = p2o3ratio_props(data["pred_P2O3"], binwidth=binwidth)
            x = np.arange(len(p2o3ratio_plot))

            ax.bar(x - bar_width + (bar_width)*run, p2o3ratio_plot, width=bar_width, color=color, hatch=hatch, edgecolor=lighten_color(color, 0.3), label=f"Run {run+1}")
            ax.set_title(f"{kernel} + {acqf}")
            ax.set_ylim(0.0, 10.5)
            ax.legend()

            ax.set_ylabel("Number of Candidates")
            p2o3_xticklabels(x, binwidth)
            ax.set_xlim(-0.5, len(p2o3ratio_plot) - 1.5)

            # add annotations and make square
            ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[(i*3)+j] + ")", transform=ax.transAxes, size=16, weight='bold')
            ax.set_box_aspect(1)

save_fig(fig3, name=f"GP_pred_plot", folder=f"active_learning_model/active_learning_model_expr")

# AGGREGATE PLOT ==================

fig4, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 26))
axes = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]]

for i in range(len(kernels)):
    kernel = kernels[i]
    color = colors[0]
    hatch = hatches[0]

    for j in range(len(acqfs)):
        ax = axes[i][j]
        acqf = acqfs[j]
        binwidth = 0.2
        bar_width=0.4 

        p2o3ratio_plot = np.zeros(8)

        for run in range(0, 3):
            data = pd.read_csv(f"active_learning_model/active_learning_model_expr/{kernel}/{kernel}_whole_dataset_{acqf}_bulk_candidates_run{run+1}.csv")
            p2o3ratio_plot = p2o3ratio_plot+ p2o3ratio_props(data["pred_P2O3"], binwidth=binwidth)

        x = np.arange(len(p2o3ratio_plot))

        ax.bar(x, p2o3ratio_plot, width=bar_width, color=color, hatch=hatch, edgecolor=lighten_color(color, 0.3))
        ax.set_title(f"{kernel} + {acqf}")
        ax.set_ylim(0.0, 30.5)
        
        ax.set_ylabel("Number of Candidates")
        p2o3_xticklabels(x, binwidth)
        ax.set_xlim(-0.5, len(p2o3ratio_plot) - 1.5)

        # add annotations and make square
        ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[(i*3)+j] + ")", transform=ax.transAxes, size=16, weight='bold')
        ax.set_box_aspect(1)

save_fig(fig4, name=f"GP_pred_plot_aggregate", folder=f"active_learning_model/active_learning_model_expr")