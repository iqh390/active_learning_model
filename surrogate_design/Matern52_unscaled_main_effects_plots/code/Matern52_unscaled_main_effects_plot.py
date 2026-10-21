# Finalised?

import os
import matplotlib.pyplot as plt
import string
import pandas as pd

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

    output_path = os.path.join("active_learning_model/", f"{folder}{name}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Saved {name}.")

plot_settings()


# 1: TIME PLOTS ===================================================

# load data
Time_05Na = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_time_0.5Na_data.csv")
Time_075Na = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_time_0.75Na_data.csv")
Time_10Na = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_time_1.0Na_data.csv")

# full plot
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (22, 10))

# add annotations
axes = [ax1, ax2, ax3]
for i in range(len(axes)):
    ax = axes[i]
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

ax1.plot(Time_05Na["Time"], 
            Time_05Na["P2O3ratio"],
            color="#FFD92F",
            linewidth=3,
            linestyle="-")
# shades 2std above and below the mean
ax1.fill_between(
    Time_05Na["Time"],
    Time_05Na["P2O3ratio"] - 2 * Time_05Na["P2O3std"],
    Time_05Na["P2O3ratio"] + 2 * Time_05Na["P2O3std"],
    color="#FFD92F",
    alpha=0.3
)
ax1.set_xlabel("Time (h)  (0.5 Na)")
ax1.set_ylabel("Predicted P2/O3")
ax1.set_ylim(0.2, 1.4)

ax2.plot(Time_075Na["Time"], 
            Time_075Na["P2O3ratio"],
            color="#FFD92F",
            linewidth=3,
            linestyle="-")
ax2.fill_between(
    Time_075Na["Time"],
    Time_075Na["P2O3ratio"] - 2 * Time_075Na["P2O3std"],
    Time_075Na["P2O3ratio"] + 2 * Time_075Na["P2O3std"],
    color="#FFD92F",
    alpha=0.3
)
ax2.set_xlabel("Time (h)  (0.75 Na)")
ax2.set_ylabel("Predicted P2/O3")
ax2.set_ylim(0.2, 1.4)

ax3.plot(Time_10Na["Time"], 
            Time_10Na["P2O3ratio"],
            color="#FFD92F",
            linewidth=3,
            linestyle="-")
ax3.fill_between(
    Time_10Na["Time"],
    Time_10Na["P2O3ratio"] - 2 * Time_10Na["P2O3std"],
    Time_10Na["P2O3ratio"] + 2 * Time_10Na["P2O3std"],
    color="#FFD92F",
    alpha=0.3
)
ax3.set_xlabel("Time (h)  (1.0 Na)")
ax3.set_ylabel("Predicted P2/O3")
ax3.set_ylim(0.2, 1.4)

save_fig(fig1, "Matern52_unscaled_main_effects_time_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 2: NA PLOTS ===============================================================

# load data
Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Na_3.0time_data.csv")
Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Na_6.0time_data.csv")
Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Na_12.0time_data.csv")

na_data = [Na_3h, Na_6h, Na_12h]

# full plot
fig2a, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (22, 10))
na_axes = [ax1, ax2, ax3]

for i in range(3):
    ax = na_axes[i]
    data = na_data[i]
    
    ax.plot(data["Na"], 
                data["P2O3ratio"],
                color="#66C2A5",
                linestyle="-")
    ax.fill_between(
        data["Na"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#66C2A5",
        alpha=0.3)

    ax.set_ylabel("Predicted P2/O3")
    ax.set_ylim(0.0, 1.5)

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

ax1.set_xlabel("%Na (3h)")
ax2.set_xlabel("%Na (6h)")
ax3.set_xlabel("%Na (12h)")

save_fig(fig2a, "Matern52_unscaled_main_effects_na_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# Zoomed in version just for completeness
fig2b, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (22, 10))
na_axes = [ax1, ax2, ax3]

for i in range(3):
    ax = na_axes[i]
    data = na_data[i]
    
    ax.plot(data["Na"], 
                data["P2O3ratio"],
                color="#66C2A5",
                linestyle="-")
    ax.fill_between(
        data["Na"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#66C2A5",
        alpha=0.3)

    ax.set_ylabel("Predicted P2/O3")

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

ax1.set_xlabel("%Na (3h)")
ax2.set_xlabel("%Na (6h)")
ax3.set_xlabel("%Na (12h)")

save_fig(fig2b, "Matern52_unscaled_main_effects_na_detailed_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 3: CU PLOTS ==================================================================

Cu_05Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.5Na_3.0time_data.csv")
Cu_05Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.5Na_6.0time_data.csv")
Cu_05Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.5Na_12.0time_data.csv")

Cu_075Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.75Na_3.0time_data.csv")
Cu_075Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.75Na_6.0time_data.csv")
Cu_075Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_0.75Na_12.0time_data.csv")

Cu_10Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_1.0Na_3.0time_data.csv")
Cu_10Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_1.0Na_6.0time_data.csv")
Cu_10Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Cu_1.0Na_12.0time_data.csv")

fig3, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
cu_data = [
    Cu_05Na_3h, Cu_05Na_6h, Cu_05Na_12h,
    Cu_075Na_3h, Cu_075Na_6h, Cu_075Na_12h,
    Cu_10Na_3h, Cu_10Na_6h, Cu_10Na_12h
    ]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

for i in range(9):
    ax = axes[i]
    data = cu_data[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)


    ax.plot(data["Cu"], 
                data["P2O3ratio"],
                color="#FC8D62",
                linewidth=3,
                linestyle="-")
    ax.fill_between(
        data["Cu"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#FC8D62",
        alpha=0.3)
    ax.set_xlabel(f"%Cu {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    # ax.set_ylim(0.45, 1.15)

save_fig(fig3, "Matern52_unscaled_main_effects_cu_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 4: FE PLOTS ==================================================================

Fe_05Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.5Na_3.0time_data.csv")
Fe_05Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.5Na_6.0time_data.csv")
Fe_05Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.5Na_12.0time_data.csv")

Fe_075Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.75Na_3.0time_data.csv")
Fe_075Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.75Na_6.0time_data.csv")
Fe_075Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_0.75Na_12.0time_data.csv")

Fe_10Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_1.0Na_3.0time_data.csv")
Fe_10Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_1.0Na_6.0time_data.csv")
Fe_10Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Fe_1.0Na_12.0time_data.csv")

fig4, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
fe_data = [
    Fe_05Na_3h, Fe_05Na_6h, Fe_05Na_12h,
    Fe_075Na_3h, Fe_075Na_6h, Fe_075Na_12h,
    Fe_10Na_3h, Fe_10Na_6h, Fe_10Na_12h
    ]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

for i in range(9):
    ax = axes[i]
    data = fe_data[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    ax.plot(data["Fe"], 
                data["P2O3ratio"],
                color="#8DA0CB",
                linewidth=3,
                linestyle="--")
    ax.fill_between(
        data["Fe"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#8DA0CB",
        alpha=0.3)
    ax.set_xlabel(f"%Fe {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    # ax.set_ylim(-0.25, 1.15)

save_fig(fig4, "Matern52_unscaled_main_effects_fe_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 5: MN PLOTS ==================================================================

Mn_05Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.5Na_3.0time_data.csv")
Mn_05Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.5Na_6.0time_data.csv")
Mn_05Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.5Na_12.0time_data.csv")

Mn_075Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.75Na_3.0time_data.csv")
Mn_075Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.75Na_6.0time_data.csv")
Mn_075Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_0.75Na_12.0time_data.csv")

Mn_10Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_1.0Na_3.0time_data.csv")
Mn_10Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_1.0Na_6.0time_data.csv")
Mn_10Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Mn_1.0Na_12.0time_data.csv")

fig5, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
mn_data = [
    Mn_05Na_3h, Mn_05Na_6h, Mn_05Na_12h,
    Mn_075Na_3h, Mn_075Na_6h, Mn_075Na_12h,
    Mn_10Na_3h, Mn_10Na_6h, Mn_10Na_12h
    ]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

for i in range(9):
    ax = axes[i]
    data = mn_data[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    ax.plot(data["Mn"], 
                data["P2O3ratio"],
                color="#E78AC3",
                linewidth=3,
                linestyle=":")
    ax.fill_between(
        data["Mn"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#E78AC3",
        alpha=0.3)
    ax.set_xlabel(f"%Mn {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    # ax.set_ylim(0.2, 1.2)

save_fig(fig5, "Matern52_unscaled_main_effects_mn_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 6: NI PLOTS ==================================================================

Ni_05Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.5Na_3.0time_data.csv")
Ni_05Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.5Na_6.0time_data.csv")
Ni_05Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.5Na_12.0time_data.csv")

Ni_075Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.75Na_3.0time_data.csv")
Ni_075Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.75Na_6.0time_data.csv")
Ni_075Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_0.75Na_12.0time_data.csv")

Ni_10Na_3h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_1.0Na_3.0time_data.csv")
Ni_10Na_6h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_1.0Na_6.0time_data.csv")
Ni_10Na_12h = pd.read_csv("active_learning_model/surrogate_design/Matern52_unscaled_main_effects_plots/Matern52_unscaled_main_effects_Ni_1.0Na_12.0time_data.csv")

fig6, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
ni_data = [
    Ni_05Na_3h, Ni_05Na_6h, Ni_05Na_12h,
    Ni_075Na_3h, Ni_075Na_6h, Ni_075Na_12h,
    Ni_10Na_3h, Ni_10Na_6h, Ni_10Na_12h
    ]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

for i in range(9):
    ax = axes[i]
    data = ni_data[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    ax.plot(data["Ni"], 
                data["P2O3ratio"],
                color="#A6D854",
                linewidth=3,
                linestyle="-.")
    ax.fill_between(
        data["Ni"],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color="#A6D854",
        alpha=0.3)
    ax.set_xlabel(f"%Ni {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    # ax.set_ylim(-0.2, 1.05)

save_fig(fig6, "Matern52_unscaled_main_effects_ni_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 7: Na and all TM main effects data on the same plots with same y axis scale, without uncertainty

fig7, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

tm_values = ["Cu", "Fe", "Mn", "Ni"]
tm_data = [cu_data, fe_data, mn_data, ni_data]

linestyles = ["-", "--", ":", "-."]
colors = ["#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"]

for i in range(9):
    ax = axes[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    for j in range (4):

        tm = tm_values[j]
        data = tm_data[j][i]
        linestyle = linestyles[j]
        color = colors[j]

        ax.plot(data[tm], 
                    data["P2O3ratio"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=3,
                    label=f"%{tm}")
        
    ax.set_xlabel(f"%TM {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    ax.set_ylim(-0.1, 1.1)

    ax.legend()

save_fig(fig7, "Matern52_unscaled_main_effects_all_tm_clean_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 8: Na and all TM main effects data on the same plots with same y axis scale, WITH UNCERTAINTY

fig8, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize = (22, 23))
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

xlabels = [
    "(3h / 0.5Na)", "(6h / 0.5Na)", "(12h / 0.5Na)",
    "(3h / 0.75Na)", "(6h / 0.75Na)", "(12h / 0.75Na)",
    "(3h / 1.0Na)", "(6h / 1.0Na)", "(12h / 1.0Na)"
    ]

tm_values = ["Cu", "Fe", "Mn", "Ni"]
tm_data = [cu_data, fe_data, mn_data, ni_data]

linestyles = ["-", "--", ":", "-."]
colors = ["#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"]

for i in range(9):
    ax = axes[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    for j in range (4):

        tm = tm_values[j]
        data = tm_data[j][i]
        linestyle = linestyles[j]
        color = colors[j]

        ax.plot(data[tm], 
                    data["P2O3ratio"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=3,
                    label=f"%{tm}")
        ax.fill_between(data[tm],
                        data["P2O3ratio"] - 2 * data["P2O3std"],
                        data["P2O3ratio"] + 2 * data["P2O3std"],
                        color=color,
                        alpha=0.2)
        
    ax.set_xlabel(f"%TM {xlabel}")
    ax.set_ylabel("Predicted P2/O3")
    # ax.set_ylim(-0.25, 1.2)

    ax.legend()

save_fig(fig8, "Matern52_unscaled_main_effects_all_tm_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")

# 9: All TM main effects data (6h / 0.75Na) on separate plots, WITH UNCERTAINTY

fig9, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (14, 14))

axes = [ax1, ax2, ax3, ax4]
tm_midpoint_data = [Cu_075Na_6h, Fe_075Na_6h, Mn_075Na_6h, Ni_075Na_6h]

tm_values = ["Cu", "Fe", "Mn", "Ni"]
linestyles = ["-", "--", ":", "-."]
colors = ["#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"]

xlabels = ["%Cu (6h / 0.75Na)",
           f"%Fe (6h / 0.75Na)",
           "%Mn (6h / 0.75Na)",
           "%Ni (6h / 0.75Na)"]

for i in range(4):

    tm = tm_values[i]
    linestyle = linestyles[i]
    color = colors[i]

    ax = axes[i]
    data = tm_midpoint_data[i]
    xlabel = xlabels[i]

    # add annotations
    ax.text(-0.07, 1.07, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes, size=16, weight='bold')
    ax.set_box_aspect(1)

    ax.plot(data[tm], 
                data["P2O3ratio"],
                color=color,
                linewidth=3,
                linestyle=linestyle)
    ax.fill_between(
        data[tm],
        data["P2O3ratio"] - 2 * data["P2O3std"],
        data["P2O3ratio"] + 2 * data["P2O3std"],
        color=color,
        alpha=0.3)
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel("Predicted P2/O3")

save_fig(fig9, "Matern52_unscaled_main_effects_all_tm_breakdown_plots", "surrogate_design/Matern52_unscaled_main_effects_plots")