import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from typing import Optional, Tuple, Union


FIGSIZE = (15, 5)
DPI = 150

ITR1_LABEL = "1"
ITR2_LABEL = "2"

XLABEL = "Distance (m)"
YLABEL = "Elevation (m)"

PLANNED_COLOR = "#2AC92A"
UNPLANNED_COLOR = "#D42525"
PLANNED_ALPHA = 0.5
UNPLANNED_ALPHA = 0.5

ITR1_COLOR = "#76c3ec"
ITR2_COLOR = "#e99930"

LEGEND_CIRCLE_SIZE = 0.75
LEGEND_HANDLETEXTPAD = 0.35
LEGEND_LABELSPACING = 0.4
LEGEND_BORDERPAD = 0.4
LEGEND_HANDLELENGTH = 1.0
LEGEND_COLUMNSPACING = 0.6

SAVE_PATH: Optional[str] = None   # set folder or full filename to save plots
SHOW_PLOT: bool = True           # whether to plt.show()


# LEGEND HANDLERS
class NumberedCircleHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + 0.5 * width / 6.0
        cy = ydescent + height / 2.0
        size_factor = getattr(orig_handle, "_size", 0.45)
        r = size_factor * height

        circle = mpatches.Circle((cx, cy), radius=r,
                                 transform=trans,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor='none', linewidth=0)
        num_text = plt.Text(cx, cy, str(getattr(orig_handle, "_num", "")),
                            transform=trans, color="white",
                            fontsize=fontsize * 0.9, ha="center", va="center")
        return [circle, num_text]


class SymbolCircleHandler(HandlerBase):
    def __init__(self, symbol: str, fontsize_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.fontsize_factor = fontsize_factor

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        cx = xdescent + 0.5 * width / 6.0
        cy = ydescent + height / 2.0
        size_factor = getattr(orig_handle, "_size", 0.45)
        r = size_factor * height

        circle = mpatches.Circle((cx, cy), radius=r,
                                 transform=trans,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor='none', linewidth=0)
        symbol_text = plt.Text(cx, cy, self.symbol,
                               transform=trans, color="white",
                               fontsize=fontsize * self.fontsize_factor, ha="center", va="center")
        return [circle, symbol_text]


# MAIN (simple) FUNCTION
def plot_elevation_profile(
    section_df: pd.DataFrame,
    section_name: str = ""
) -> Tuple[plt.Figure, plt.Axes, float, str]:
    """
    Plot elevation profiles and highlight planned/unplanned blocks.
    Inputs:
        section_df: DataFrame with required columns:
            - chainage, z_itr1, z_itr2, line_name, area_name
            - optional: label_name
        section_name: string used in title and saved filename.

    Returns:
        (fig, ax, max_unplanned_dev, deviation_flag) where deviation_flag is "yes"/"no".
    """
    required = {"chainage", "z_itr1", "z_itr2", "line_name", "area_name"}
    if not required.issubset(section_df.columns):
        raise ValueError(f"section_df missing required columns: {required - set(section_df.columns)}")

    # Prepare data (sort + reset index)
    df_full = section_df.sort_values("chainage").reset_index(drop=True)

    # Create plot and base profiles
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(df_full["chainage"], df_full["z_itr1"], color=ITR1_COLOR, linewidth=1.4, label=f"ITR-{ITR1_LABEL}")
    ax.plot(df_full["chainage"], df_full["z_itr2"], color=ITR2_COLOR, linewidth=1.4, label=f"ITR-{ITR2_LABEL}")

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Elevation Profile Section {section_name.replace('_', ' ')}")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.6)

    planned_set = {"planned_and_done_excavation", "planned_and_done_dump"}
    unplanned_set = {"unplanned_and_done_excavation", "unplanned_and_used_dump", "unplanned_and_done_dump"}

    n = len(df_full)
    planned_handles, planned_labels = [], []
    unplanned_handles, unplanned_labels = [], []

    # global z-range for marker offsets
    zmax = df_full[["z_itr1", "z_itr2"]].max().max()
    zmin = df_full[["z_itr1", "z_itr2"]].min().min()
    z_range = zmax - zmin if zmax != zmin else 1.0

    i = 0
    planned_num, unplanned_num = 1, 1
    marker_min = float("inf")
    marker_max = float("-inf")
    max_unplanned_dev = None
    has_unplanned = False

    # iterate consecutive blocks identified by line_name
    while i < n:
        lbl = df_full.loc[i, "line_name"]
        if pd.isna(lbl) or (lbl not in planned_set and lbl not in unplanned_set):
            i += 1
            continue

        start_i = i
        # choose label_name if present, else area_name
        label_candidate = None
        if "label_name" in df_full.columns:
            label_candidate = df_full.loc[start_i, "label_name"]
        if pd.isna(label_candidate) or label_candidate is None:
            label_candidate = df_full.loc[start_i, "area_name"]

        # advance to block end
        i += 1
        while i < n and df_full.loc[i, "line_name"] == lbl:
            i += 1
        end_i = i - 1

        x_block = df_full.loc[start_i:end_i, "chainage"].to_numpy()
        z1_block = df_full.loc[start_i:end_i, "z_itr1"].to_numpy()
        z2_block = df_full.loc[start_i:end_i, "z_itr2"].to_numpy()
        if x_block.size == 0:
            continue

        mid_chain = (float(x_block[0]) + float(x_block[-1])) / 2.0

        if lbl in planned_set:
            elev_marker = max(np.max(z1_block), np.max(z2_block)) + 0.02 * z_range
            ax.fill_between(x_block, z1_block, z2_block, color=PLANNED_COLOR, alpha=PLANNED_ALPHA, linewidth=0)
            ax.scatter([mid_chain], [elev_marker], s=220, color=PLANNED_COLOR, zorder=6, edgecolors='none')
            ax.text(mid_chain, elev_marker, str(planned_num), ha="center", va="center", fontsize=8, color="white", zorder=7)

            legend_circle = mpatches.Circle((0, 0), radius=0.5, facecolor=PLANNED_COLOR, edgecolor='none')
            legend_circle._num = planned_num
            legend_circle._size = LEGEND_CIRCLE_SIZE
            planned_handles.append(legend_circle)
            planned_labels.append(str(label_candidate) if label_candidate is not None else "")
            planned_num += 1
            marker_max = max(marker_max, elev_marker)

        elif lbl in unplanned_set:
            deviation = float(np.max(np.abs(z1_block - z2_block)))
            if max_unplanned_dev is None or deviation > max_unplanned_dev:
                max_unplanned_dev = deviation
            has_unplanned = True

            elev_marker = min(np.min(z1_block), np.min(z2_block)) - 0.02 * z_range
            ax.fill_between(x_block, z1_block, z2_block, color=UNPLANNED_COLOR, alpha=UNPLANNED_ALPHA, linewidth=0)
            ax.scatter([mid_chain], [elev_marker], s=220, color=UNPLANNED_COLOR, zorder=6, edgecolors='none')
            ax.text(mid_chain, elev_marker, str(unplanned_num), ha="center", va="center", fontsize=8, color="white", zorder=7)

            legend_circle = mpatches.Circle((0, 0), radius=0.5, facecolor=UNPLANNED_COLOR, edgecolor='none')
            legend_circle._num = unplanned_num
            legend_circle._size = LEGEND_CIRCLE_SIZE
            unplanned_handles.append(legend_circle)

            label_text = (str(label_candidate) if label_candidate is not None else "")
            unplanned_labels.append(f"{label_text} ({deviation:.2f} m)")
            unplanned_num += 1
            marker_min = min(marker_min, elev_marker)

    # add side legends for numbered blocks
    if planned_handles:
        handler_map = {h: NumberedCircleHandler() for h in planned_handles}
        leg_planned = ax.legend(handles=planned_handles, labels=planned_labels,
                                loc="lower left", bbox_to_anchor=(0.02, 0.02),
                                frameon=True, fontsize="small", handler_map=handler_map,
                                handletextpad=LEGEND_HANDLETEXTPAD,
                                labelspacing=LEGEND_LABELSPACING,
                                borderpad=LEGEND_BORDERPAD,
                                handlelength=LEGEND_HANDLELENGTH)
        ax.add_artist(leg_planned)

    if unplanned_handles:
        handler_map = {h: NumberedCircleHandler() for h in unplanned_handles}
        ax.legend(handles=unplanned_handles, labels=unplanned_labels,
                  loc="lower right", bbox_to_anchor=(0.98, 0.02),
                  frameon=True, fontsize="small", handler_map=handler_map,
                  handletextpad=LEGEND_HANDLETEXTPAD,
                  labelspacing=LEGEND_LABELSPACING,
                  borderpad=LEGEND_BORDERPAD,
                  handlelength=LEGEND_HANDLELENGTH)

    # main bottom legend
    planned_symbol = mpatches.Circle((0, 0), radius=0.5, facecolor=PLANNED_COLOR, edgecolor='none')
    planned_symbol._size = LEGEND_CIRCLE_SIZE
    unplanned_symbol = mpatches.Circle((0, 0), radius=0.5, facecolor=UNPLANNED_COLOR, edgecolor='none')
    unplanned_symbol._size = LEGEND_CIRCLE_SIZE

    itr1_handle = Line2D([0], [0], color=ITR1_COLOR, lw=2)
    itr2_handle = Line2D([0], [0], color=ITR2_COLOR, lw=2)
    proxy_dev = Line2D([], [], linestyle="", marker=None, color="none")

    dev_label = f"Deviation detected: {max_unplanned_dev:.2f} m" if has_unplanned and max_unplanned_dev is not None else "No deviation detected"

    fig.legend(
        handles=[planned_symbol, unplanned_symbol, itr1_handle, itr2_handle, proxy_dev],
        labels=["Planned Area", "Unplanned Area", f" ITR {ITR1_LABEL}", f" ITR {ITR2_LABEL}", dev_label],
        loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=5, frameon=False,
        fontsize="small",
        handler_map={planned_symbol: SymbolCircleHandler("✓"), unplanned_symbol: SymbolCircleHandler("✗")},
        handletextpad=0.0000005,
    )

    # endpoint labels (use df_full)
    if section_name and "_" in section_name:
        left_label, right_label = section_name.split("_", 1)
        x_start = float(df_full["chainage"].iloc[0])
        x_end = float(df_full["chainage"].iloc[-1])
        y_start = np.nanmean([df_full["z_itr1"].iloc[0], df_full["z_itr2"].iloc[0]])
        y_end = np.nanmean([df_full["z_itr1"].iloc[-1], df_full["z_itr2"].iloc[-1]])
        offset = 0.03 * z_range
        ax.text(x_start, y_start + offset, left_label, ha="left", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x_end,   y_end + offset,   right_label, ha="right", va="bottom", fontsize=10, fontweight="bold")

    # expand y-limits if markers were placed
    if marker_min != float("inf") or marker_max != float("-inf"):
        cur_ylim = ax.get_ylim()
        margin = 0.03 * z_range
        new_ymin = min(cur_ylim[0], marker_min - margin) if marker_min != float("inf") else cur_ylim[0]
        new_ymax = max(cur_ylim[1], marker_max + margin) if marker_max != float("-inf") else cur_ylim[1]
        ax.set_ylim(new_ymin, new_ymax)

    plt.subplots_adjust(bottom=0.18)

    # # save if requested
    # if SAVE_PATH:
    #     save_target = SAVE_PATH
    #     if os.path.isdir(save_target) or save_target.endswith(os.sep):
    #         safe_section = section_name.replace("_", " ")
    #         filename = f"elevation_profile_{safe_section}.png"
    #         save_target = os.path.join(save_target, filename)
    #     fig.savefig(save_target, dpi=DPI, bbox_inches="tight")

    # return normalized outputs
    if not has_unplanned or max_unplanned_dev is None:
        return fig, ax, 0.0, "no"
    else:
        return fig, ax, float(max_unplanned_dev), "yes"
