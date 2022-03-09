#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:14:07 2022

@author: moritz
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
import cmcrameri
import pandas as pd
import seaborn as sns
import networkx as nx
from statsmodels.stats.weightstats import DescrStatsW

plt.style.use("seaborn")
plt.rcParams.update({"axes.facecolor": "white"})

# %% Fig. 1
arrowsizevar = 20
widthvar = 3
node_sizevar = 5000
alphavar = 0.3
font_sizevar = 35

pos = {r"$z$": (1 / 2, np.sqrt(3 / 4)), r"$y$": (0, 0), r"$x$": (1, 0)}

# fig. a)
fig, ax = plt.subplots()
ax.set_xlim(ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2)
ax.set_ylim(ax.get_ylim()[0] - 0.2, ax.get_ylim()[1] + 0.1)
G = nx.DiGraph()
G.add_edge(r"$z$", r"$x$")
G.add_edge(r"$y$", r"$y$")  # dummy line, otherwise the y node is not plotted
nx.draw_networkx_edges(
    G,
    # edgelist = list(G.edges())[0],
    pos=pos,
    node_size=node_sizevar,
    alpha=1,
    width=widthvar,
)

# draw white circles over the lines
nx.draw_networkx_nodes(
    G,
    pos=pos,
    node_size=node_sizevar,
    alpha=1,
    node_color="w",
)
# draw bigger white circle to hide dummy line
nx.draw_networkx_nodes(
    G,
    nodelist=[r"$y$"],
    pos=pos,
    node_size=node_sizevar * 5,
    alpha=1,
    node_color="w",
)
# draw the nodes as desired
nx.draw_networkx_nodes(
    G,
    pos=pos,
    node_size=node_sizevar,
    alpha=alphavar,
)
nx.draw_networkx_labels(G, pos=pos, font_size=font_sizevar)
plt.axis("off")
plt.savefig("plots/fig1_a.png", dpi=300)
plt.show()
plt.close()

# fig. b)
fig, ax = plt.subplots()
ax.set_xlim(ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2)
ax.set_ylim(ax.get_ylim()[0] - 0.2, ax.get_ylim()[1] + 0.1)
H = nx.DiGraph()
H.add_edge(r"$z$", r"$y$")
H.add_edge(r"$y$", r"$x$")
nx.draw_networkx_edges(H, pos=pos, node_size=node_sizevar, alpha=1, width=widthvar)
# add indirect edge (dashed)
ax.annotate(
    "",
    xy=(1, 0),
    xycoords="data",
    xytext=(1 / 2, np.sqrt(3 / 4)),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->",
        ls="dashed",
        lw=widthvar,
        connectionstyle="arc3",
        shrinkA=38,
        shrinkB=38,
    ),
)
# draw white circles over the lines
nx.draw_networkx_nodes(
    H,
    pos=pos,
    node_size=node_sizevar,
    alpha=1,
    node_color="w",
)
# draw the nodes as desired
nx.draw_networkx_nodes(
    H,
    pos=pos,
    node_size=node_sizevar,
    alpha=alphavar,
)
nx.draw_networkx_labels(H, pos=pos, font_size=font_sizevar)
plt.axis("off")
plt.savefig("plots/fig1_b.png", dpi=300)
plt.close()


# fig. c)
fig, ax = plt.subplots()
ax.set_xlim(ax.get_xlim()[0] - 0.2, ax.get_xlim()[1] + 0.2)
ax.set_ylim(ax.get_ylim()[0] - 0.2, ax.get_ylim()[1] + 0.1)
J = nx.DiGraph()
J.add_edge(r"$y$", r"$z$")
J.add_edge(r"$y$", r"$x$")
nx.draw_networkx_edges(J, pos=pos, node_size=node_sizevar, alpha=1, width=widthvar)
# add indirect edge (dashed)
ax.annotate(
    "",
    xy=(1, 0),
    xycoords="data",
    xytext=(1 / 2, np.sqrt(3 / 4)),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->",
        ls="dashed",
        lw=widthvar,
        connectionstyle="arc3",
        shrinkA=38,
        shrinkB=38,
    ),
)
# annotate lags
ax.annotate(
    r"$\mathcal{L}^\alpha$",
    xy=(0.25, 0.5),
    xycoords="data",
    xytext=(0.15, 0.5),
    textcoords="data",
    fontsize=25,
)
ax.annotate(
    r"$\mathcal{L}^\beta$",
    xy=(0.5, 0),
    xycoords="data",
    xytext=(0.45, -0.2),
    textcoords="data",
    fontsize=25,
)
ax.annotate(
    r"$\alpha < \beta$",
    xy=(0.5, 0.5 * np.sqrt(3 / 4)),
    xycoords="data",
    xytext=(0.35, 0.3 * np.sqrt(3 / 4)),
    textcoords="data",
    fontsize=25,
)
# draw white circles over the lines
nx.draw_networkx_nodes(
    J,
    pos=pos,
    node_size=node_sizevar,
    alpha=1,
    node_color="w",
)
# draw the nodes as desired
nx.draw_networkx_nodes(
    J,
    pos=pos,
    node_size=node_sizevar,
    alpha=alphavar,
)
nx.draw_networkx_labels(J, pos=pos, font_size=font_sizevar)
plt.axis("off")
plt.savefig("plots/fig1_c.png", dpi=300)
plt.close()

# %% Fig. 2
tests = [
    "G-causality",
    "PRSA_KS_onesided",
    "PRSA_AD_onesided",
    "PRSA_KS_twosided",
    "PRSA_SW",
]

testnames = [
    "Granger-causality",
    "BPRSA: Kolmogorov-Smirnov one-sided",
    "BPRSA: Anderson-Darling",
    "BPRSA: Kolmogorov-Smirnov two-sided",
    "BPRSA: Shapiro-Wilk",
]  # for creating the diagram titles
colors = [cmcrameri.cm.imolaS(ii) for ii in [0, 2, 17, 1, 3]]
fig, ax = plt.subplots(figsize=(6, 6))
# colors = [cmcrameri.cm.imola(int((ii)*255/(len(tests)))) for ii in range(len(tests))]
# colors = [cmcrameri.cm.imolaS(ii + 33) for ii in range(len(tests))]
# colors = [cm.get_cmap("inferno", (len(tests)+1))(ii) for ii in range(len(tests))]
handles = []  # for legend
labels = []  # for legend
for aa in [1, 4, 2, 0]:  # adjust the order so that legend fits sequence of lines
    handles.append(mlines.Line2D([], [], color=colors[aa], lw=5))
    labels.append(testnames[aa])
    resultarray = pd.read_csv(
        os.path.join("csv", f"{tests[aa]}_x_sample-size.csv"),
        dtype=float,
        header=0,
        index_col=0,
    )
    # resultarray = [
    #     [float(resultarray[ll][mm]) for mm in range(len(resultarray[ll]))]
    #     for ll in range(len(resultarray))
    # ]
    if aa == 2:
        # AD-Test: where are 50 % of the samples identified as causal?
        iso = [0.5]
    else:
        # other tests: where is the average p-value equal to 5 %?
        iso = [0.05]
    ax.contour(
        range(11),
        range(100),
        resultarray,
        levels=iso,
        colors=[colors[aa]],
        linestyles="solid",
        linewidths=5,
    )
ax.set_xlabel("sample size n", fontsize=20)
ax.set_ylabel(r"q (influence of $z$ on $x$)", fontsize=20)
ax.set_xticklabels(
    [
        "64",
        "256",
        "1024",
        "4096",
        "16384",
        "65536",
    ],
    fontsize=16,
)
ax.tick_params(axis="both", which="both", direction="out", length=8, width=1)
xticks_major = range(0, 11, 2)
yticks = [20, 48, 60, 88, 100]
ax.set_yticks(yticks)
ax.set_yticklabels([0.01, 0.05, 0.1, 0.5, 1], fontsize=16)
for tick in yticks:
    ax.axhline(tick, color="#EEEEEE", lw=1, zorder=1)
for tick in xticks_major:
    ax.axvline(tick, color="#EEEEEE", lw=1, zorder=1)
# fit line
ax.axline([6, 48], slope=-5.7, color="#555555", lw=4, ls="dashed")
ax.annotate(
    r"$q \sim n^{-0.5}$",
    # r"$q \cdot n^{0.5} = 3.2$",
    [2, 45],
    color="#555555",
    fontsize=15,
)
# create legend
legend = plt.legend(
    handles=handles,
    labels=labels,
    loc="lower left",
    fontsize=14,
    title="p = 0.05 isolines",
    title_fontproperties={"size": 15, "weight": "bold"},
)
legend._legend_box.align = "left"
plt.tight_layout()
plt.savefig("./plots/fig2.png".format(tests[aa]), dpi=300)
plt.show()
plt.close()

# %% Fig. 3

colors_list = [cmcrameri.cm.imolaS(ii) for ii in [0, 2, 17]]
ax.set_xlim(5, 25)
ax.set_ylim(5, 25)
for REVERSE in [True, False]:
    fig, ax = plt.subplots(figsize=(6, 6))
    # colors = [cmcrameri.cm.imolaS(ii) for ii in range(2)]
    for mode, colors, levels in zip(
        ["without_s1", "with_s1"],
        [colors_list[:2], colors_list[2:]],
        [[0, 0.05, 1], [0, 0.05]],
    ):
        # if mode =="without_s1":
        #     continue
        if REVERSE:
            resultarray = pd.read_csv(
                os.path.join("csv", f"test_G_causality_x2_x3_{mode}.csv"),
                header=0,
                index_col=0,
            )
        else:
            resultarray = pd.read_csv(
                os.path.join("csv", f"test_G_causality_x2_x3_{mode}.csv"),
                header=0,
                index_col=0,
            )
        # resultarray = [
        #     [float(resultarray[ll][mm]) for mm in range(len(resultarray[ll]))]
        #     for ll in range(len(resultarray))
        # ]
        ax.contourf(
            range(len(resultarray)),
            range(len(resultarray)),
            resultarray,
            levels=levels,
            colors=colors,
            alpha=0.8,
        )
        ax.contour(
            range(len(resultarray)),
            range(len(resultarray)),
            resultarray,
            levels=[0.05, 1],
            colors="black",
            linewidths=5,
        )
    xticks = [5, 12, 15, 22, 25]
    yticks = [5, 12, 15, 22, 25]
    ax.set_xticks(xticks)
    ax.set_xticklabels([0.01, 0.05, 0.1, 0.5, 1], fontsize=15)
    ax.set_yticks(yticks)
    ax.set_yticklabels([0.01, 0.05, 0.1, 0.5, 1], fontsize=15)

    ax.set_xlabel(r"$q_{y\rightarrow z}$ (influence of $y$ on $z$)", fontsize=16)
    ax.set_ylabel(r"$q_{y\rightarrow x}$ (influence of $y$ on $x$)", fontsize=16)
    ax.text(19, 8, "Region 1", fontsize=15, fontweight="bold", ha="center")
    ax.text(19, 4, "no link\nidentified\n(False)", fontsize=15, ha="center")
    ax.text(22, 16.5, "Region 2", fontsize=15, fontweight="bold", ha="center")
    ax.text(22, 12.5, "indirect link\nidentified\n (True)", fontsize=15, ha="center")
    ax.text(21, 23.5, "Region 3", fontsize=15, fontweight="bold", ha="center")
    ax.text(21, 19.5, "direct link\nidentified\n (False)", fontsize=15, ha="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # for tick in yticks:
    #     ax.axhline(tick, color="white", lw=1.5, zorder=4)
    # for tick in xticks:
    #     ax.axvline(tick, color="white", lw=2, zorder=4)
    ax.tick_params(axis="both", which="both", direction="out", length=8, width=1)
    plt.tight_layout()
    # plot fit line
    ax.axline([16, 16], slope=-1, color="white", lw=5, ls="dashed")
    ax.annotate(
        r"$q_{y\rightarrow x} * q_{y\rightarrow z}$ = 0.02",
        [2, 19],
        color="white",
        fontsize=15,
    )
    if REVERSE:
        plt.savefig("./plots/fig3_reverse.png", dpi=300)
    else:
        plt.savefig("./plots/fig3.png", dpi=300)
    plt.show()
    plt.close()

# %% Fig. 4
group = "YC"
subject = "B000102"
times = [1000, 1181]  # seconds, start and end of plotted range
signals = ["heart", "breath", "alpha"]
signal_names = [
    "Heart\nrate\n[Hz]",
    "Breathing\nrate\n[Hz]",
    "EEG\nα amplitude\n[μV]",
]
resolutions = [1, 2, 5, 10]  # seconds, must be integers
colors = [cm.viridis(ii / len(resolutions)) for ii in range(len(resolutions))]

file_path = os.path.join(
    "csv", "timeseries_resampled", group, f"timeseries_resampled_1hz_{subject}.csv"
)
df = pd.read_csv(file_path, header=0, index_col=0)

fig, ax = plt.subplots(3, figsize=(6, 4), sharex=True)
for resolution, color in zip(resolutions, colors):
    sl = df.loc[times[0] : times[1]]
    time_interp = [
        np.mean(sl.loc[jj : jj + resolution - 1].index)
        for jj in range(times[0], times[1], resolution)
    ]
    for ii, (signal, signal_name) in enumerate(zip(signals, signal_names)):
        s_interp = [
            np.mean(sl[signal].loc[jj : jj + resolution - 1])
            for jj in range(times[0], times[1], resolution)
        ]
        ax[ii].plot(time_interp, s_interp, color=color, label=f"{resolution} s")
        ax[ii].tick_params(axis="y", which="both", direction="out", length=8, width=1)
        ax[ii].set_ylabel(signal_name, rotation=0, labelpad=30, va="center")
ax[-1].set_xticks(range(times[0], times[1], 60))
ax[-1].set_xticklabels(range((times[1] - times[0]) // 60 + 1))
ax[-1].tick_params(axis="x", which="both", direction="out", length=8, width=1)
ax[0].legend(
    bbox_to_anchor=[1, 0.3],
    fontsize=14,
    title="Resolution",
    title_fontproperties={"size": 15, "weight": "bold"},
)
fig.align_ylabels()
fig.supxlabel("Time [min]")
plt.savefig("plots/fig4.png", dpi=300, bbox_inches="tight")
plt.close()

# %% Fig. 5
group = "YC"
model_lag_order = 5
band = "alpha"
# 'YC', 'EC', 'OSA', 'surrogate'
# - Young Control, Elderly Control, Sleep Apnea, surrogate
surrogate = False
# this is just to make sure all files will be named correctly. A bit dirty.
if surrogate:
    surmarker_f = "surrogate/"
    surmarker_n = "surrogate_"
else:
    surmarker_f = ""
    surmarker_n = ""
colors = [cmcrameri.cm.imolaS(ii) for ii in [0, 2, 17, 3]]
timescales = [1, 2, 5, 10, 15, 30, 60, 120]  # seconds

for group in ["YC"]:
    stat_df_new = pd.read_csv(
        f"./csv/stat_df_new_{surmarker_n}{group}_lo{model_lag_order}_{band}.csv",
        header=0,
        index_col=0,
    )
    stat_df_new = stat_df_new * 100
    fig, ax = plt.subplots()
    g = sns.barplot(
        data=stat_df_new.drop(labels="120", axis="columns")
        .drop(labels="awake", axis="rows")
        .unstack()
        .reset_index(),
        x="level_0",
        y=0,
        hue="level_1",
        ax=ax,
        palette=colors,
    )
    for tick in (20, 40, 60, 80):
        g.axhline(tick, color="#EEEEEE", lw=1, zorder=0.9)
    g.set_facecolor("white")
    ax.set_xlabel("time series resolution [s]")
    ax.set_ylabel("stationary time [%]")
    leg = ax.legend(facecolor="white", frameon=None)
    leg.get_frame().set_linewidth(0.0)
    plt.savefig(
        "./plots/fig5.png".format(surmarker_n, group, model_lag_order, band),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()

# %% Fig. 6
# define helper function for plotting
def get_row_for_plotting(s1, s2):
    """Helper function that decides which row a signal pair (s1, s2) should
    be plotted in."""

    pair = [s1, s2]
    if "breath" in pair and "heart" in pair:
        return 0
    elif "breath" in pairs[jj] and "eeg_amp_o1" in pairs[jj]:
        return 1
    elif "heart" in pairs[jj] and "eeg_amp_o1" in pairs[jj]:
        return 2
    else:
        return ValueError("Not a permissible combination of signal names")


# constants for plotting
PLT_UPPER_BOUND = 0.1
PLT_LOWER_BOUND = -0.015
PLT_XTICKS = [1, 5, 10, 15]
PLT_YTICKS = [0, 0.04, 0.08]
FS = 14
SURROGATE = False
SURMARKER_N = "surrogate_" if SURROGATE else ""
sleepstages = ["LS", "DS", "REM"]
sleepstages_names = ["Light Sleep", "Deep Sleep", "REM"]
sleepstages_dic = dict(zip(sleepstages, sleepstages_names))
for group in ["YC", "EC", "OSA"]:
    averages_df = pd.read_csv(
        os.path.join("csv", f"avg_df_{surmarker_n}{group}_{band}.csv"),  # dtype=float,
        header=[0, 1],
        index_col=[0, 1],
    )
    st_err_df = pd.read_csv(
        os.path.join(
            "csv", f"st_err_df_{surmarker_n}{group}_{band}.csv"
        ),  # dtype=float,
        header=[0, 1],
        index_col=[0, 1],
    )
    # averages_df = averages_df.astype(float)
    # st_err_df = st_err_df.astype(float)

    # set missing data points to NaN so the plotting works properly
    for ii in averages_df.index:
        for jj in averages_df.columns:
            if averages_df.loc[ii, jj].size == 0:
                averages_df.loc[ii, jj] = np.nan
    for ii in st_err_df.index:
        for jj in st_err_df.columns:
            if st_err_df.loc[ii, jj].size == 0:
                st_err_df.loc[ii, jj] = np.nan
    # 6. plot G as function of time scale, one (sub-)plot for every sleep stage
    colorlist = ("mediumpurple", "midnightblue", "red", "darkred", "gold", "#FFAA00")
    # pairs = [(caused, causing)]
    pairs = [
        ("breath", "heart"),
        ("breath", "eeg_amp_o1"),
        ("heart", "breath"),
        ("heart", "eeg_amp_o1"),
        ("eeg_amp_o1", "breath"),
        ("eeg_amp_o1", "heart"),
    ]

    fig, axes = plt.subplots(
        3, len(sleepstages), sharex="col", sharey="row", figsize=[8, 7]
    )
    for ii in range(len(sleepstages)):
        for jj in range(len(pairs)):
            sns.lineplot(
                data=averages_df.unstack(level=1),
                x=averages_df.unstack(level=1).index,
                y=(*pairs[jj], sleepstages[ii]),
                color=colorlist[jj],
                ax=axes[get_row_for_plotting(*pairs[jj]), ii],
            )
            axes[get_row_for_plotting(*pairs[jj]), ii].errorbar(
                x=averages_df.unstack(level=1).index,
                y=(averages_df.unstack(level=1)[(*pairs[jj], sleepstages[ii])]),
                color=colorlist[jj],
                yerr=st_err_df.unstack(level=1)[(*pairs[jj], sleepstages[ii])],
                fmt="o",
            )
            if ii > 0:
                sns.despine(left=True)
                # axes[ii].set_yticklabels([])
        for jj in range(len(axes)):
            if jj == 0:
                axes[jj, ii].set_title(sleepstages_dic[sleepstages[ii]], fontsize=FS)
            axes[jj, ii].set_ylim(PLT_LOWER_BOUND, PLT_UPPER_BOUND)
            axes[jj, ii].set_xscale("linear")
            axes[jj, ii].set_ylabel("")
            axes[jj, ii].set_xticks(PLT_XTICKS)
            axes[jj, ii].set_xticklabels(PLT_XTICKS, fontsize=15)
            axes[jj, ii].set_yticks(PLT_YTICKS)
            axes[jj, ii].set_yticklabels(PLT_YTICKS, fontsize=15)
            axes[jj, ii].set_xlabel("")
            for tick in PLT_YTICKS:
                axes[jj, ii].axhline(tick, color="#EEEEEE", lw=1, zorder=1)
    fig.supxlabel("time series resolution [s]", fontsize=FS)
    fig.supylabel("G", fontsize=FS, rotation=0)
    # add legend to YC plot
    if group == "YC" and not SURROGATE:
        axes[0, 1].text(5, 0.065, "H → B", fontsize=FS + 3, color=colorlist[0])
        axes[0, 1].text(5, 0.04, "B → H", fontsize=FS + 3, color=colorlist[2])
        axes[1, 1].text(5, 0.065, "α → B", fontsize=FS + 3, color=colorlist[1])
        axes[1, 1].text(5, 0.04, "B → α", fontsize=FS + 3, color=colorlist[4])
        axes[2, 1].text(5, 0.065, "α → H", fontsize=FS + 3, color=colorlist[3])
        axes[2, 1].text(5, 0.04, "H → α", fontsize=FS + 3, color=colorlist[5])
    plt.savefig(
        "./plots/fig6_{}{}.png".format(group, SURMARKER_N),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()

# %% Fig. 7
SURROGATE = False
groups = ["YC", "EC", "OSA"]
MODEL_LAG_ORDER = 5
WIDTHVAR_O = 100
NODE_SIZEVAR = 5000
ALPHAVAR = 0.5
FONT_SIZEVAR = 30
sleepstages = ["LS", "DS", "REM"]
sleepstages_names = ["Light Sleep", "Deep Sleep", "REM"]
for group in groups:
    if SURROGATE and group != "YC":
        continue
    nodelabels = {"breath": "B", "heart": "H", "eeg_amp_o1": r"$\alpha$"}
    G_df = pd.read_csv(
        os.path.join(
            "csv",
            "G_df",
            f"G_df_{SURMARKER_N}{group}_lo{MODEL_LAG_ORDER}_complete.csv",
        ),
        header=[0, 1],
        index_col=[0, 1],
    )
    averages_df = pd.read_csv(
        os.path.join("csv", f"avg_df_{SURMARKER_N}{group}_{band}.csv"),  # dtype=float,
        header=[0, 1],
        index_col=[0, 1],
    )

    pos = {
        "breath": (1 / 2, np.sqrt(3 / 4)),
        "heart": (0, 0),
        "eeg_amp_o1": (1, 0),
    }

    fig, axes = plt.subplots(1, len(sleepstages), figsize=[15, 4], tight_layout=True)
    for ii in range(len(sleepstages)):
        axes[ii].set_title(sleepstages_names[ii], fontsize=35)
        axes[ii].set_aspect("equal")
        axes[ii].set_xlim(axes[ii].get_xlim()[0] - 0.2, axes[ii].get_xlim()[1] + 0.2)
        axes[ii].set_ylim(axes[ii].get_ylim()[0] - 0.2, axes[ii].get_ylim()[1] + 0.1)
        J = nx.Graph()
        nx.draw_networkx_nodes(
            J, pos=pos, node_size=NODE_SIZEVAR, nodelist=pos.keys(), ax=axes[ii]
        )

        # add edges 'manually'
        for pair in pairs:
            widthvar = WIDTHVAR_O * np.mean(
                # take the value for time series resolution 10 seconds
                averages_df.loc[(10, sleepstages[ii]), pair]
            )
            axes[ii].annotate(
                "",
                xy=pos[pair[0]],
                xycoords="data",
                xytext=pos[pair[1]],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    ls="solid",
                    lw=widthvar,
                    color="black",
                    connectionstyle="arc3,rad=0.1",
                    mutation_scale=20,
                    shrinkA=35,
                    shrinkB=35,
                ),
            )

        # draw white circles over the lines
        nx.draw_networkx_nodes(
            J,
            pos=pos,
            label=None,
            node_size=NODE_SIZEVAR,
            alpha=1,
            node_color="w",
            nodelist=pos.keys(),
            ax=axes[ii],
        )
        # draw the nodes as desired
        nx.draw_networkx_nodes(
            J,
            pos=pos,
            node_size=NODE_SIZEVAR,
            alpha=ALPHAVAR,
            nodelist=pos.keys(),
            ax=axes[ii],
        )
        nx.draw_networkx_labels(
            J,
            pos=pos,
            font_size=FONT_SIZEVAR,
            ax=axes[ii],
            labels=nodelabels,
        )
        axes[ii].axis("off")

    plt.savefig(os.path.join("plots", f"fig7_{SURMARKER_N}{group}.png"), dpi=300)
    plt.show()
    plt.close()

# %% plot legend
fig_legend, ax_legend = plt.subplots(figsize=(15, 1.5))
ax_legend.axis("off")
pos_legend = {
    "p1": (0, 0),
    "p2": (1 / 6, 0),
    "p3": (2 / 6, 0),
    "p4": (3 / 6, 0),
    "p5": (4 / 6, 0),
    "p6": (5 / 6, 0),
}
pairs_legend = [("p1", "p2"), ("p3", "p4"), ("p5", "p6")]
width_vars_legend = [ii * WIDTHVAR_O for ii in [0.01, 0.04, 0.07]]
L = nx.Graph()

for pair, widthvar in zip(pairs_legend, width_vars_legend):
    ax_legend.annotate(
        "",
        xy=pos_legend[pair[0]],
        xycoords="data",
        xytext=pos_legend[pair[1]],
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            ls="solid",
            lw=widthvar,
            color="black",
            mutation_scale=20,
        ),
    )
    ax_legend.annotate(
        f"G = {widthvar / WIDTHVAR_O}",
        (np.mean([pos_legend[pair[0]][0], pos_legend[pair[1]][0]]), 0.5),
        xycoords="axes fraction",
        fontsize=30,
        va="center",
        ha="center",
    )
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 0.1)
plt.savefig("./plots/fig7_legend.png", dpi=300)
plt.show()
plt.close()
