# Finalised

# plots lengthscales from kernel_hyperparameters_v2.csv

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Testing hyperparam data types to plot ===============================
hyperparams = pd.read_csv("active_learning_model/surrogate_design/kernel_testing/kernel_hyperparameters_v3.csv")

# checking data types
print(type(hyperparams["kernel_name"][0]))
print(type(hyperparams["noise_constraint"][0]))
print(type(hyperparams["outputscale"][2]))
print(type(hyperparams["noise"][0]))
print(type(hyperparams["lengthscale"][1]))

# New csv for lengthscales only for easier data handling ==================

data = pd.read_csv("active_learning_model/surrogate_design/kernel_testing/kernel_hyperparams_lengthscales_ard_v3.csv")

# Plotting grouped bar chart of lengthscales for each kernel ==============

features = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"] # features are set in this order

# Only take data from scales kernels with ARD applied, low default noise
rbf = data[data["kernel_name"] == "Scaled RBF ARD Low Noise"]
matern32 = data[data["kernel_name"] == "Scaled Matern 32 ARD Low Noise"]
matern52 = data[data["kernel_name"] == "Scaled Matern 52 ARD Low Noise"]

# Isolate features
rbf = rbf[features].values.reshape(-1)
matern32 = matern32[features].values.reshape(-1)
matern52 = matern52[features].values.reshape(-1)

# List of indices in order of lowest to highest lengthscale for RBF
sorted_indices = np.argsort(rbf).tolist()
print(sorted_indices)

print(type(rbf))
print(rbf)

# np.argsort()

# Bar height and x locations
h, y = 0.25, np.arange(len(features))

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

fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(y + h, rbf[sorted_indices], height=h, color="#1f77b4", hatch='\\', edgecolor=lighten_color("#1f77b4", 0.3), label="RBF") # horizontal bar chart
ax.barh(y, matern32[sorted_indices], height=h, color="#ff7f0e", hatch='//', edgecolor=lighten_color("#ff7f0e", 0.3), label="Matern 32")
ax.barh(y - h, matern52[sorted_indices], height=h, color="#2ca02c", hatch="xx", edgecolor=lighten_color("#2ca02c", 0.3), label="Matern 52")

ax.set_yticks(y)
ax.set_yticklabels([features[i] for i in sorted_indices]) # sort features manually 
ax.set_xlabel("Relative Length Scale")
ax.set_ylabel("Feature")
ax.legend(fontsize=14)

save_fig(fig, name="kernel_hyperparams_plot", folder="surrogate_design/kernel_testing")
