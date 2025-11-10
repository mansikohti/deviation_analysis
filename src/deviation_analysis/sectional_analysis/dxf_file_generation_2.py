import math
import re
import pandas as pd
import ezdxf
from pathlib import Path
from ezdxf import colors as ezcolors, units
from ezdxf.lldxf import const as ezconst

# ------------------------
# GLOBAL / DEFAULT SETTINGS (edit these instead of function params)
# ------------------------
ITER1_COLOR_HEX = "#1f77b4"
ITER2_COLOR_HEX = "#ff7f0e"
LAYER_ITER1 = "ITR_2024_Profile"
LAYER_ITER2 = "ITR_2025_Profile"

# text / axes
TEXT_LAYER = "Text"
TEXT_HEIGHT = 2.0
SHOW_AXES = True
AXIS_LAYER = "Axes"
AXIS_COLOR_HEX = "#000000"
X_LABEL = "Chainage (m)"
Y_LABEL = "Elevation (m)"
TARGET_TICK_COUNT = 7
TICK_LEN_FACTOR = 0.6
LABEL_GAP_FACTOR = 0.5
Y_TICK_LABEL_SIDE_FACTOR = 2.5

# transforms
SCALE_X = 1.0
SCALE_Y = 1.0
OFFSET_X = 0.0
OFFSET_Y = 0.0

# title
TITLE = "Elevation Profile (m)"

# ---- FIXED column names (do not change inside function call)
CHAIN_COL = "chainage"
ELEV1_COL = "z_itr1"
ELEV2_COL = "z_itr2"
LINE_NAME_COL = "line_name"
LABEL_NAME_COL = "label_name"

# fills
LAYER_FILL_PLANNED = "Fill_Planned"
LAYER_FILL_UNPLANNED = "Fill_Unplanned"
PLANNED_HEX = "#27ae60"
UNPLANNED_HEX = "#c0392b"

# badges
ADD_ZONE_BADGES = True
LABEL_LAYER = "Zone_Badges"
BADGE_RADIUS_FACTOR = 0.9
BADGE_POSITION = "auto"  
BADGE_OFFSET_FACTOR = 1.6

# section labels
SECTION_LABEL_LAYER = "Section_Labels"
SECTION_OFFSET_FACTOR = 2.0
SECTION_X_PAD_FACTOR = 1.5

# legends
SHOW_LEGENDS = True
LEGEND_LAYER = "Legend"
LEGEND_PAD_FACTOR = 1.0
LEGEND_ROW_GAP_FACTOR = 0.5
LEGEND_BULLET_RADIUS_FACTOR = 0.5
LEGEND_COL_GAP_FACTOR = 0.7
BOTTOM_LEGEND_GAP_FACTOR = 5
BOTTOM_LEGEND_COL_GAP_FACTOR = 10.0

# ------------------------
# Function (only df/path, out_path, section_name are inputs)
# ------------------------
def export_elevation_profiles_to_dxf(
    df,
    out_path: str,
    section_name: str | None = None,
) -> str:
    """
    Build and save DXF elevation profile.
    Inputs:
      - df : pandas.DataFrame or path-to-csv (string)
      - out_path : where to save the DXF
      - section_name : optional section label (e.g. "K_K'")
    Columns expected (fixed globals):
      CHAIN_COL, ELEV1_COL, ELEV2_COL, LINE_NAME_COL (see module globals)
    Returns: absolute path to saved DXF file.
    """

    # ---------- helpers (copied, using globals)
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError(f"Bad hex color: {h}")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _nice_step(span: float, target: int) -> float:
        if span <= 0:
            return 1.0
        step = span / max(1, target)
        mag = 10 ** math.floor(math.log10(step))
        norm = step / mag
        return (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10) * mag

    def _format_tick(v: float) -> str:
        a = abs(v)
        if a >= 1000 or a == 0:
            return f"{v:.0f}"
        if a >= 100:
            return f"{v:.1f}"
        if a >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def _add_profile(msp, df_local: pd.DataFrame, ccol: str, zcol: str, layer: str, rgb: tuple[int, int, int]):
        d = df_local[[ccol, zcol]].dropna().sort_values(ccol)
        if d.empty:
            return
        pts = [
            (float(r[ccol]) * SCALE_X + OFFSET_X, float(r[zcol]) * SCALE_Y + OFFSET_Y)
            for _, r in d.iterrows()
        ]
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer, "true_color": ezcolors.rgb2int(rgb)})

    def _add_text_centered(s, x, y, rgb=None, h=None, layer=None):
        t = msp.add_text(
            s,
            dxfattribs={
                "layer": layer or TEXT_LAYER,
                "height": h or TEXT_HEIGHT,
                "halign": ezconst.CENTER,
                "valign": ezconst.MIDDLE,
                **({"true_color": ezcolors.rgb2int(rgb)} if rgb else {}),
            },
        )
        t.dxf.insert = (x, y)
        t.dxf.align_point = (x, y)
        return t

    def _add_text_left(s, x, y, h=None, layer=None):
        t = msp.add_text(
            s,
            dxfattribs={
                "layer": layer or TEXT_LAYER,
                "height": h or TEXT_HEIGHT,
                "halign": ezconst.LEFT,
                "valign": ezconst.MIDDLE,
            },
        )
        t.dxf.insert = (x, y)
        t.dxf.align_point = (x, y)
        return t

    def _first_non_empty(series, fallback: str) -> str:
        if series is None:
            return fallback
        for v in series:
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    return s
        return fallback

    # ---------- load df if path provided
    if isinstance(df, (str, Path)):
        df = pd.read_csv(str(df), low_memory=False)

    # check required columns exist
    for col in (CHAIN_COL, ELEV1_COL, ELEV2_COL):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # ---------- create dxf document
    doc = ezdxf.new("R2018", units=units.M)
    doc.header["$INSUNITS"] = units.M
    for layer in (
        LAYER_ITER1,
        LAYER_ITER2,
        TEXT_LAYER,
        AXIS_LAYER if SHOW_AXES else None,
        LAYER_FILL_PLANNED,
        LAYER_FILL_UNPLANNED,
        LABEL_LAYER if ADD_ZONE_BADGES else None,
        SECTION_LABEL_LAYER if section_name else None,
        LEGEND_LAYER if SHOW_LEGENDS else None,
    ):
        if layer and layer not in doc.layers:
            doc.layers.add(layer)
    msp = doc.modelspace()

    # profiles
    _add_profile(msp, df, CHAIN_COL, ELEV1_COL, LAYER_ITER1, _hex_to_rgb(ITER1_COLOR_HEX))
    _add_profile(msp, df, CHAIN_COL, ELEV2_COL, LAYER_ITER2, _hex_to_rgb(ITER2_COLOR_HEX))

    # title
    if TITLE:
        t = msp.add_text(TITLE, dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT})
        t.dxf.insert = (OFFSET_X, OFFSET_Y + TEXT_HEIGHT * 4)

    # bounding vars for legends
    plot_x_min = plot_x_max = plot_y_min = plot_y_max = None

    # axes
    if SHOW_AXES:
        chains = df[CHAIN_COL].dropna().astype(float)
        elevs = pd.concat(
            [df[ELEV1_COL].dropna().astype(float), df[ELEV2_COL].dropna().astype(float)]
        )
        ch_min, ch_max = float(chains.min()), float(chains.max())
        z_min, z_max = float(elevs.min()), float(elevs.max())
        xs, ys = _nice_step(ch_max - ch_min, TARGET_TICK_COUNT), _nice_step(
            z_max - z_min, TARGET_TICK_COUNT
        )

        x0 = math.floor(ch_min / xs) * xs
        x1 = math.ceil(ch_max / xs) * xs
        y0 = math.floor(z_min / ys) * ys
        y1 = math.ceil(z_max / ys) * ys

        ox = x0 * SCALE_X + OFFSET_X
        oy = y0 * SCALE_Y + OFFSET_Y
        axis_rgb = _hex_to_rgb(AXIS_COLOR_HEX)
        tick_len = TICK_LEN_FACTOR * TEXT_HEIGHT
        gap = LABEL_GAP_FACTOR * TEXT_HEIGHT

        plot_x_min = ox
        plot_x_max = x1 * SCALE_X + OFFSET_X
        plot_y_min = oy
        plot_y_max = y1 * SCALE_Y + OFFSET_Y

        # axis lines
        msp.add_line(
            (ox, oy), (x1 * SCALE_X + OFFSET_X, oy),
            dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(axis_rgb)},
        )
        msp.add_line(
            (ox, oy), (ox, y1 * SCALE_Y + OFFSET_Y),
            dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(axis_rgb)},
        )

        # X ticks & labels
        xt = x0
        while xt <= x1 + 1e-9:
            x_pos = xt * SCALE_X + OFFSET_X
            msp.add_line(
                (x_pos, oy), (x_pos, oy - tick_len),
                dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(axis_rgb)},
            )
            tx = msp.add_text(_format_tick(xt), dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT})
            tx.dxf.insert = (x_pos, oy - tick_len - gap)
            xt += xs

        # Y ticks & labels
        yt = y0
        while yt <= y1 + 1e-9:
            y_pos = yt * SCALE_Y + OFFSET_Y
            msp.add_line(
                (ox, y_pos), (ox - tick_len, y_pos),
                dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(axis_rgb)},
            )
            ty = msp.add_text(
                _format_tick(yt),
                dxfattribs={
                    "layer": TEXT_LAYER,
                    "height": TEXT_HEIGHT,
                    "halign": ezconst.RIGHT,
                    "valign": ezconst.MIDDLE,
                },
            )
            extra = Y_TICK_LABEL_SIDE_FACTOR * TEXT_HEIGHT
            anchor_x = ox - tick_len - gap - extra
            ty.dxf.insert = (anchor_x, y_pos)
            ty.dxf.align_point = (anchor_x, y_pos)
            yt += ys

        # axis labels
        xtxt = msp.add_text(X_LABEL, dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT})
        xtxt.dxf.insert = ((ox + (x1 * SCALE_X + OFFSET_X)) / 2, oy - 3 * TEXT_HEIGHT)

        ytxt = msp.add_text(
            Y_LABEL, dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT, "rotation": 90},
        )
        ytxt.dxf.insert = (ox - 4 * TEXT_HEIGHT, (oy + (y1 * SCALE_Y + OFFSET_Y)) / 2)

    # fills + badges + legend capture (unplanned ALWAYS drawn)
    legend_items_planned = []   # (idx, label)
    legend_items_unplanned = [] # (idx, label, deviation)
    max_unplanned_dev = 0.0

    if LINE_NAME_COL in df.columns:
        valid = {
            "planned_and_done_excavation",
            "unplanned_and_done_excavation",
            "planned_and_done_dump",
            "unplanned_and_done_dump",
        }
        ln = df[LINE_NAME_COL].where(df[LINE_NAME_COL].isin(valid))
        grp = (ln.ne(ln.shift()) & ln.notna()).cumsum()
        segs = df.loc[ln.notna()].assign(_grp=grp[ln.notna()].values)

        badge_r = BADGE_RADIUS_FACTOR * TEXT_HEIGHT
        planned_idx = 1
        unplanned_idx = 1

        grouped = list(segs.groupby("_grp"))
        grouped.sort(key=lambda g: float(g[1][CHAIN_COL].astype(float).min()))

        for _, seg in grouped:
            name = seg[LINE_NAME_COL].iloc[0]

            sub = (
                seg[[CHAIN_COL, ELEV1_COL, ELEV2_COL]]
                .dropna(subset=[CHAIN_COL, ELEV1_COL, ELEV2_COL])
                .sort_values(CHAIN_COL)
            )

            label_text = _first_non_empty(seg.get(LABEL_NAME_COL), name)

            if len(sub) < 2:
                if name.startswith("planned"):
                    legend_items_planned.append((planned_idx, label_text))
                    planned_idx += 1
                else:
                    # short unplanned: choose to add minimal legend entry
                    legend_items_unplanned.append((unplanned_idx, label_text, 0.0))
                    unplanned_idx += 1
                continue

            is_planned = name.startswith("planned")
            hatch_layer = LAYER_FILL_PLANNED if is_planned else LAYER_FILL_UNPLANNED
            rgb = _hex_to_rgb(PLANNED_HEX if is_planned else UNPLANNED_HEX)

            p_top = [
                (float(r[CHAIN_COL]) * SCALE_X + OFFSET_X, float(r[ELEV1_COL]) * SCALE_Y + OFFSET_Y)
                for _, r in sub.iterrows()
            ]
            p_bot = [
                (float(r[CHAIN_COL]) * SCALE_X + OFFSET_X, float(r[ELEV2_COL]) * SCALE_Y + OFFSET_Y)
                for _, r in sub.iloc[::-1].iterrows()
            ]
            boundary = p_top + p_bot

            dev = float((sub[ELEV1_COL].astype(float) - sub[ELEV2_COL].astype(float)).abs().max())
            if not is_planned:
                max_unplanned_dev = max(max_unplanned_dev, dev)

            # always hatch (planned & unplanned)
            hatch = msp.add_hatch(
                dxfattribs={"layer": hatch_layer, "true_color": ezcolors.rgb2int(rgb)}
            )
            hatch.set_solid_fill()
            hatch.paths.add_polyline_path(boundary, is_closed=True)

            # mid-point for badge
            mid_i = len(sub) // 2
            ch_mid = float(sub.iloc[mid_i][CHAIN_COL])
            z1_mid = float(sub.iloc[mid_i][ELEV1_COL])
            z2_mid = float(sub.iloc[mid_i][ELEV2_COL])
            top_z = max(z1_mid, z2_mid)
            bot_z = min(z1_mid, z2_mid)

            pos = BADGE_POSITION.lower()
            if pos == "auto":
                pos = "top" if is_planned else "bottom"
            if pos == "top":
                z_label = top_z + BADGE_OFFSET_FACTOR * (TEXT_HEIGHT / max(1e-9, SCALE_Y))
            else:
                z_label = bot_z - BADGE_OFFSET_FACTOR * (TEXT_HEIGHT / max(1e-9, SCALE_Y))

            bx = ch_mid * SCALE_X + OFFSET_X
            by = z_label * SCALE_Y + OFFSET_Y

            if is_planned:
                idx = planned_idx
                planned_idx += 1
                legend_items_planned.append((idx, label_text))
            else:
                idx = unplanned_idx
                unplanned_idx += 1
                legend_items_unplanned.append((idx, label_text, dev))

            if ADD_ZONE_BADGES:
                msp.add_circle((bx, by), badge_r,
                               dxfattribs={"layer": LABEL_LAYER, "true_color": ezcolors.rgb2int(rgb)})
                tt = msp.add_text(
                    str(idx),
                    dxfattribs={
                        "layer": LABEL_LAYER,
                        "height": TEXT_HEIGHT,
                        "true_color": ezcolors.rgb2int(rgb),
                        "halign": ezconst.CENTER,
                        "valign": ezconst.MIDDLE,
                    },
                )
                tt.dxf.insert = (bx, by)
                tt.dxf.align_point = (bx, by)

    # section end labels
    if section_name:
        try:
            left_tag, right_tag = [s.strip() for s in section_name.split("_", 1)]
        except ValueError:
            left_tag = section_name.strip()
            right_tag = ""

        d2 = df[[CHAIN_COL, ELEV1_COL, ELEV2_COL]].dropna().astype(float).sort_values(CHAIN_COL)
        if len(d2) >= 1:
            ch_min = float(d2[CHAIN_COL].min())
            ch_max = float(d2[CHAIN_COL].max())

            pad_ch = SECTION_X_PAD_FACTOR * (TEXT_HEIGHT / max(1e-9, SCALE_X))

            i_left = (d2[CHAIN_COL] - ch_min).abs().idxmin()
            i_right = (d2[CHAIN_COL] - ch_max).abs().idxmin()
            z_left = max(float(d2.loc[i_left, ELEV1_COL]), float(d2.loc[i_left, ELEV2_COL]))
            z_right = max(float(d2.loc[i_right, ELEV1_COL]), float(d2.loc[i_right, ELEV2_COL]))

            x_l = (ch_min + pad_ch) * SCALE_X + OFFSET_X
            x_r = (ch_max - pad_ch) * SCALE_X + OFFSET_X
            y_l = z_left * SCALE_Y + OFFSET_Y + SECTION_OFFSET_FACTOR * TEXT_HEIGHT
            y_r = z_right * SCALE_Y + OFFSET_Y + SECTION_OFFSET_FACTOR * TEXT_HEIGHT

            if left_tag:
                tl = msp.add_text(left_tag, dxfattribs={"layer": SECTION_LABEL_LAYER, "height": TEXT_HEIGHT})
                tl.dxf.insert = (x_l, y_l)

            if right_tag:
                tr = msp.add_text(right_tag, dxfattribs={"layer": SECTION_LABEL_LAYER, "height": TEXT_HEIGHT})
                tr.dxf.insert = (x_r, y_r)

    # legends render
    if SHOW_LEGENDS and (plot_x_min is not None):
        rgb_plan = _hex_to_rgb(PLANNED_HEX)
        rgb_unpl = _hex_to_rgb(UNPLANNED_HEX)
        rgb_itr1 = _hex_to_rgb(ITER1_COLOR_HEX)
        rgb_itr2 = _hex_to_rgb(ITER2_COLOR_HEX)

        pad = LEGEND_PAD_FACTOR * TEXT_HEIGHT
        row_gap = LEGEND_ROW_GAP_FACTOR * TEXT_HEIGHT
        bullet_r = LEGEND_BULLET_RADIUS_FACTOR * TEXT_HEIGHT
        col_gap = LEGEND_COL_GAP_FACTOR * TEXT_HEIGHT

        # LEFT list: planned
        xL = plot_x_min + pad
        total_rows = len(legend_items_planned)
        y_start = plot_y_min + pad + (total_rows - 1) * (TEXT_HEIGHT + row_gap)
        for idx, lbl in sorted(legend_items_planned, key=lambda t: t[0]):
            msp.add_circle((xL, y_start), bullet_r,
                        dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_plan)})
            _add_text_centered(str(idx), xL, y_start, rgb=rgb_plan, layer=LEGEND_LAYER)
            _add_text_left(str(lbl), xL + bullet_r + col_gap, y_start, layer=LEGEND_LAYER)
            y_start -= TEXT_HEIGHT + row_gap

        # RIGHT list: unplanned
        xR = plot_x_max - pad
        total_rows = len(legend_items_unplanned)
        y_start = plot_y_min + pad + (total_rows - 1) * (TEXT_HEIGHT + row_gap)
        for idx, lbl, dev in sorted(legend_items_unplanned, key=lambda t: t[0]):
            msp.add_circle((xR, y_start), bullet_r,
                        dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_unpl)})
            _add_text_centered(str(idx), xR, y_start, rgb=rgb_unpl, layer=LEGEND_LAYER)
            lab = f"{lbl} ({dev:.2f} m)"
            t = msp.add_text(
                lab, dxfattribs={"layer": LEGEND_LAYER, "height": TEXT_HEIGHT,
                                "halign": ezconst.RIGHT, "valign": ezconst.MIDDLE}
            )
            tx = xR - (bullet_r + col_gap)
            t.dxf.insert = (tx, y_start); t.dxf.align_point = (tx, y_start)
            y_start -= TEXT_HEIGHT + row_gap

        # BOTTOM center strip
        yB = plot_y_min - BOTTOM_LEGEND_GAP_FACTOR * TEXT_HEIGHT
        cx = (plot_x_min + plot_x_max) / 2.0
        dx = BOTTOM_LEGEND_COL_GAP_FACTOR * TEXT_HEIGHT

        x = cx - 2 * dx
        msp.add_circle((x, yB), bullet_r,
                       dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_plan)})
        _add_text_centered("✓", x, yB, rgb=rgb_plan, h=TEXT_HEIGHT*0.95, layer=LEGEND_LAYER)
        _add_text_left("Planned Area", x + bullet_r + col_gap, yB, layer=LEGEND_LAYER)

        x = cx - dx
        msp.add_circle((x, yB), bullet_r,
                       dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_unpl)})
        _add_text_centered("✗", x, yB, rgb=rgb_unpl, h=TEXT_HEIGHT*0.95, layer=LEGEND_LAYER)
        _add_text_left("Unplanned Area", x + bullet_r + col_gap, yB, layer=LEGEND_LAYER)

        x = cx + 0
        msp.add_line((x - 1.8*TEXT_HEIGHT, yB), (x + 1.8*TEXT_HEIGHT, yB),
                     dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_itr1)})
        _add_text_left("ITR 2024", x + 2.0*TEXT_HEIGHT, yB, layer=LEGEND_LAYER)

        x = cx + dx
        msp.add_line((x - 1.8*TEXT_HEIGHT, yB), (x + 1.8*TEXT_HEIGHT, yB),
                     dxfattribs={"layer": LEGEND_LAYER, "true_color": ezcolors.rgb2int(rgb_itr2)})
        _add_text_left("ITR 2025", x + 2.0*TEXT_HEIGHT, yB, layer=LEGEND_LAYER)

        x = cx + 2 * dx
        dev_lbl = (
            f"Deviation detected: {max_unplanned_dev:.2f} m"
            if legend_items_unplanned else "No deviation detected"
        )
        _add_text_left(dev_lbl, x, yB, layer=LEGEND_LAYER)

    # save
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_p)
    return str(out_p.resolve())


# example usage (replace paths as needed)
if __name__ == "__main__":
    df_path = r"D:/2_Analytics/6_plan_vs_actual/10_nov_output_1/sectional_analysis/E_E'/section_data.csv"
    out_dxf = r"D:/2_Analytics/6_plan_vs_actual/10_nov_output_1/sectional_analysis/E_E'/dxf_file.dxf"
    export_elevation_profiles_to_dxf(df_path, out_dxf, section_name="E_E'")
