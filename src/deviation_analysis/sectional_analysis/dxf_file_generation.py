# dxf_builder_fixed.py
import re
import math
from pathlib import Path
import pandas as pd
import ezdxf
from ezdxf import colors as ezcolors
from ezdxf import units
from ezdxf.document import Drawing   # correct import for type-hinting

# -------------------------
# Reduced PARAMETERS (module-level)
# -------------------------
ITER1_COLOR_HEX = "#1f77b4"
ITER2_COLOR_HEX = "#ff7f0e"
PLANNED_HEX = "#27ae60"
UNPLANNED_HEX = "#c0392b"

LAYER_ITER1 = "ITR_2024_Profile"
LAYER_ITER2 = "ITR_2025_Profile"
LAYER_FILL_PLANNED = "Fill_Planned"
LAYER_FILL_UNPLANNED = "Fill_Unplanned"
TEXT_LAYER = "Text"
LABEL_LAYER = "Zone_Badges"
LEGEND_LAYER = "Legend"
SECTION_LABEL_LAYER = "Section_Labels"
AXIS_LAYER = "Axes"

TEXT_HEIGHT = 2.0
SHOW_AXES = True
TITLE = "Elevation Profile (m)"

CHAIN_COL = "chainage"
ELEV1_COL = "z_itr1"
ELEV2_COL = "z_itr2"
LINE_NAME_COL = "line_name"
LABEL_NAME_COL = "label_name"

ADD_ZONE_BADGES = True
SHOW_LEGENDS = True

# -------------------------
# small helpers
# -------------------------
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Bad hex color: {h}")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _first_non_empty(series, fallback: str) -> str:
    if series is None:
        return fallback
    for v in series:
        if pd.notna(v):
            s = str(v).strip()
            if s and s.lower() != "nan":
                return s
    return fallback

def _nice_step(span: float, target: int = 7) -> float:
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

# -------------------------
# Main builder (returns ezdxf Drawing; does not save)
# -------------------------
def build_elevation_profile_dxf_from_df(df: pd.DataFrame, section_name: str | None = None) -> Drawing:
    """
    Build and return an ezdxf Drawing object for the elevation profile.
    Does NOT save the DXF — caller is responsible for saving if needed.
    Assumes columns: CHAIN_COL, ELEV1_COL, ELEV2_COL exist (defaults above).
    """
    # detect columns (use defaults if present)
    chain_col = CHAIN_COL if CHAIN_COL in df.columns else next((c for c in df.columns if re.search(r"chain", c, re.I)), None)
    elev1_col = ELEV1_COL if ELEV1_COL in df.columns else next((c for c in df.columns if re.search(r"z_itr1|elev.*1|z1", c, re.I)), None)
    elev2_col = ELEV2_COL if ELEV2_COL in df.columns else next((c for c in df.columns if re.search(r"z_itr2|elev.*2|z2", c, re.I)), None)

    if chain_col is None or elev1_col is None or elev2_col is None:
        raise ValueError("Required columns not found (chainage, two elevation columns).")

    # create doc
    doc = ezdxf.new("R2018", units=units.M)
    doc.header["$INSUNITS"] = units.M
    for L in (LAYER_ITER1, LAYER_ITER2, TEXT_LAYER, AXIS_LAYER, LAYER_FILL_PLANNED, LAYER_FILL_UNPLANNED, LABEL_LAYER, SECTION_LABEL_LAYER, LEGEND_LAYER):
        if L and L not in doc.layers:
            doc.layers.add(L)
    msp = doc.modelspace()

    # draw two profiles (simple polyline)
    def _add_profile(ccol, zcol, layer, color_hex):
        d = df[[ccol, zcol]].dropna().sort_values(ccol)
        if d.empty:
            return
        pts = [(float(r[ccol]), float(r[zcol])) for _, r in d.iterrows()]
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer, "true_color": ezcolors.rgb2int(_hex_to_rgb(color_hex))})

    _add_profile(chain_col, elev1_col, LAYER_ITER1, ITER1_COLOR_HEX)
    _add_profile(chain_col, elev2_col, LAYER_ITER2, ITER2_COLOR_HEX)

    # simple title
    if TITLE:
        t = msp.add_text(TITLE, dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT})
        min_chain = float(df[chain_col].min())
        max_elev = float(df[[elev1_col, elev2_col]].max().max())
        t.dxf.insert = (min_chain, max_elev + 4 * TEXT_HEIGHT)

    # axes
    if SHOW_AXES:
        chains = df[chain_col].dropna().astype(float)
        elevs = pd.concat([df[elev1_col].dropna().astype(float), df[elev2_col].dropna().astype(float)])
        ch_min, ch_max = float(chains.min()), float(chains.max())
        z_min, z_max = float(elevs.min()), float(elevs.max())

        xs = _nice_step(ch_max - ch_min, 7)
        ys = _nice_step(z_max - z_min, 7)

        x0 = math.floor(ch_min / xs) * xs
        x1 = math.ceil(ch_max / xs) * xs
        y0 = math.floor(z_min / ys) * ys
        y1 = math.ceil(z_max / ys) * ys

        msp.add_line((x0, y0), (x1, y0), dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(_hex_to_rgb("#000000"))})
        msp.add_line((x0, y0), (x0, y1), dxfattribs={"layer": AXIS_LAYER, "true_color": ezcolors.rgb2int(_hex_to_rgb("#000000"))})

        # ticks/labels simplified
        xt = x0
        while xt <= x1 + 1e-9:
            msp.add_line((xt, y0), (xt, y0 - 0.6 * TEXT_HEIGHT), dxfattribs={"layer": AXIS_LAYER})
            tx = msp.add_text(_format_tick(xt), dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT})
            tx.dxf.insert = (xt, y0 - 1.0 * TEXT_HEIGHT)
            xt += xs

        yt = y0
        while yt <= y1 + 1e-9:
            msp.add_line((x0, yt), (x0 - 0.6 * TEXT_HEIGHT, yt), dxfattribs={"layer": AXIS_LAYER})
            ty = msp.add_text(_format_tick(yt), dxfattribs={"layer": TEXT_LAYER, "height": TEXT_HEIGHT, "halign": "right"})
            ty.dxf.insert = (x0 - 1.5 * TEXT_HEIGHT, yt)
            yt += ys

    # fills + badges
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

        planned_idx = 1
        unplanned_idx = 1
        badge_r = 0.9 * TEXT_HEIGHT

        for _, seg in segs.groupby("_grp"):
            name = seg[LINE_NAME_COL].iloc[0]
            sub = seg[[chain_col, elev1_col, elev2_col]].dropna().sort_values(chain_col)
            if len(sub) < 2:
                continue
            is_planned = name.startswith("planned")
            rgb_hex = PLANNED_HEX if is_planned else UNPLANNED_HEX
            p_top = [(float(r[chain_col]), float(r[elev1_col])) for _, r in sub.iterrows()]
            p_bot = [(float(r[chain_col]), float(r[elev2_col])) for _, r in sub.iloc[::-1].iterrows()]
            boundary = p_top + p_bot
            hatch = msp.add_hatch(dxfattribs={"layer": LAYER_FILL_PLANNED if is_planned else LAYER_FILL_UNPLANNED, "true_color": ezcolors.rgb2int(_hex_to_rgb(rgb_hex))})
            hatch.set_solid_fill()
            hatch.paths.add_polyline_path(boundary, is_closed=True)

            if ADD_ZONE_BADGES:
                mid = sub.iloc[len(sub) // 2]
                ch_mid = float(mid[chain_col])
                z_mid_top = max(float(mid[elev1_col]), float(mid[elev2_col]))
                z_label = z_mid_top + 1.6 * TEXT_HEIGHT if is_planned else min(float(mid[elev1_col]), float(mid[elev2_col])) - 1.6 * TEXT_HEIGHT
                idx = planned_idx if is_planned else unplanned_idx
                if is_planned:
                    planned_idx += 1
                else:
                    unplanned_idx += 1
                msp.add_circle((ch_mid, z_label), badge_r, dxfattribs={"layer": LABEL_LAYER, "true_color": ezcolors.rgb2int(_hex_to_rgb(rgb_hex))})
                tt = msp.add_text(str(idx), dxfattribs={"layer": LABEL_LAYER, "height": TEXT_HEIGHT, "halign": "center", "valign": "middle", "true_color": ezcolors.rgb2int(_hex_to_rgb(rgb_hex))})
                tt.dxf.insert = (ch_mid, z_label)
                tt.dxf.align_point = (ch_mid, z_label)

    # section labels
    if section_name:
        try:
            left_tag, right_tag = [s.strip() for s in section_name.split("_", 1)]
        except ValueError:
            left_tag = section_name.strip()
            right_tag = ""
        d2 = df[[chain_col, elev1_col, elev2_col]].dropna().astype(float).sort_values(chain_col)
        if len(d2) >= 1:
            ch_min = float(d2[chain_col].min())
            ch_max = float(d2[chain_col].max())
            i_left = (d2[chain_col] - ch_min).abs().idxmin()
            i_right = (d2[chain_col] - ch_max).abs().idxmin()
            z_left = max(float(d2.loc[i_left, elev1_col]), float(d2.loc[i_left, elev2_col]))
            z_right = max(float(d2.loc[i_right, elev1_col]), float(d2.loc[i_right, elev2_col]))
            offset_v = 2.0 * TEXT_HEIGHT
            if left_tag:
                t = msp.add_text(left_tag, dxfattribs={"layer": SECTION_LABEL_LAYER, "height": TEXT_HEIGHT})
                t.dxf.insert = (ch_min, z_left + offset_v)
            if right_tag:
                t = msp.add_text(right_tag, dxfattribs={"layer": SECTION_LABEL_LAYER, "height": TEXT_HEIGHT})
                t.dxf.insert = (ch_max, z_right + offset_v)

    return doc

# Example usage (comment out when used as module)
if __name__ == "__main__":
    df_path = r"D:/2_Analytics/6_plan_vs_actual/10_nov_output_1/sectional_analysis/E_E'/section_data.csv"
    df = pd.read_csv(df_path)
    doc = build_elevation_profile_dxf_from_df(df, section_name="E_E'")
    doc.saveas("D:/2_Analytics/6_plan_vs_actual/10_nov_output_1/sectional_analysis/E_E'/E_E_profile.dxf")   
