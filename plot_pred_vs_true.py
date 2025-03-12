import os
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_density_plots(dist_data, out_file, title_suffix=""):
    sns.set_theme(style="whitegrid")
    pts_sorted = sorted(dist_data.keys())

    num_plots = len(pts_sorted)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))

    fig_width = max(5 * cols, 8)
    fig_height = 5 * rows
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_width, fig_height), constrained_layout=True
    )

    fig.suptitle(f"Density Plots {title_suffix}", fontsize=16)

    if num_plots == 1:
        axes = np.array([axes])

    cat_labels = ["Unstim", "Stim True", "Stim Pred"]
    cat_colors = ["blue", "red", "green"]

    for i, (pt, ax) in enumerate(zip(pts_sorted, axes.flatten())):
        for label, color in zip(cat_labels, cat_colors):
            arr = dist_data[pt][label]
            if arr.size > 0:
                sns.kdeplot(
                    arr,
                    ax=ax,
                    label=f"{label} (n={arr.size})",
                    color=color,
                    fill=False,  # set tot True to fill the area under the curve
                    alpha=0.3,
                )

        ax.set_title(f"Patient: {pt}", fontsize=14)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
    for j in range(i + 1, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def plot_result(prediction_path, original_path, marker, outdir_path):
    target = ad.read(original_path)
    target1 = target[:, marker].copy()
    stim = pd.Series(
        target1[target1.obs["condition"] == "stim"].X.flatten(), name="Stim True"
    )
    unstim = pd.Series(
        target1[target1.obs["condition"] == "control"].X.flatten(), name="Unstim"
    )
    dataf = pd.read_csv(prediction_path)

    dist_data = {
        "Patient_1": {
            "Stim True": stim.values,
            "Stim Pred": dataf["Stim Pred"].values,
            "Unstim": unstim.values,
        }
    }

    create_density_plots(dist_data, outdir_path, title_suffix="")
    return
