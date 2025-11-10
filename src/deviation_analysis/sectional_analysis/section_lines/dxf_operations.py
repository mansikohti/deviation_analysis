import os
import ezdxf
import geopandas as gpd
from collections import defaultdict, Counter
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree


def get_layers_from_dxf(input_dxf, output_folder):
    """Extract section lines layer from DXF file based on specific criteria."""
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, "section_lines.dxf")

    ALLOWED_TYPES = {"LINE", "LWPOLYLINE", "TEXT", "MTEXT"}
    DISALLOWED_TYPES = {"HATCH", "POLYLINE"}

    def is_lwpolyline_closed(lp):
        """Check if LWPOLYLINE is closed."""
        closed_attr = getattr(lp, "closed", None)
        if isinstance(closed_attr, bool):
            return closed_attr
        try:
            flags = int(lp.dxf.flags)
            return bool(flags & 1)
        except Exception:
            return True

    def layer_matches_section_criteria(entities):
        """Validate layer contains only allowed types with text and lines."""
        if not entities:
            return False

        type_counts = Counter(e.dxftype() for e in entities)

        if any(t in DISALLOWED_TYPES for t in type_counts):
            return False

        if not set(type_counts).issubset(ALLOWED_TYPES):
            return False

        text_count = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)
        if text_count == 0:
            return False

        lineish_count = type_counts.get("LINE", 0) + type_counts.get("LWPOLYLINE", 0)
        if lineish_count == 0:
            return False

        for e in entities:
            if e.dxftype() == "LWPOLYLINE" and is_lwpolyline_closed(e):
                return False

        return True

    def save_entities_as_dxf(entities, filename):
        """Save entities to new DXF file."""
        new_doc = ezdxf.new(setup=True)
        new_msp = new_doc.modelspace()
        for e in entities:
            try:
                new_msp.add_foreign_entity(e)
            except Exception as ex:
                print(f"  Skipped {e.dxftype()}: {ex}")
        new_doc.saveas(filename)

    try:
        doc = ezdxf.readfile(input_dxf)
    except Exception as e:
        print(f" Failed to read DXF: {e}")
        return "process not completed"

    msp = doc.modelspace()
    layer_entities = defaultdict(list)
    
    for ent in msp:
        layer_entities[ent.dxf.layer].append(ent)

    candidates = []
    for layer, ents in layer_entities.items():
        if layer_matches_section_criteria(ents):
            tcount = sum(1 for e in ents if e.dxftype() in ("TEXT", "MTEXT"))
            total = len(ents)
            candidates.append((layer, tcount, total, ents))

    if not candidates:
        print(" No layer matched the 'section lines + letters' criteria.")
        return "no section lines found"

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_layer, best_texts, best_total, best_entities = candidates[0]

    save_entities_as_dxf(best_entities, out_file)

    print(f" Selected layer: '{best_layer}'")
    print(f"   - Text count: {best_texts}")
    print(f"   - Entity total: {best_total}")
    
    type_counts = Counter(e.dxftype() for e in best_entities)
    for t, c in sorted(type_counts.items()):
        print(f"   - {t}: {c}")
    print(f" Saved as: {out_file}")

    return out_file


def clean_section_lines_dxf(section_lines_dxf, output_folder, prec=6):
    """Deduplicate LINE, LWPOLYLINE, TEXT, and MTEXT entities in DXF."""
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "section_lines_clean.dxf")

    def qr(val, p=prec):
        """Round value to precision."""
        return round(float(val), p)

    def qpt3(v, p=prec):
        """Convert point to rounded (x,y,z) tuple."""
        if hasattr(v, "x"):
            return (qr(v.x, p), qr(v.y, p), qr(getattr(v, "z", 0.0), p))
        if len(v) == 2:
            return (qr(v[0], p), qr(v[1], p), 0.0)
        return (qr(v[0], p), qr(v[1], p), qr(v[2], p))

    doc_in = ezdxf.readfile(section_lines_dxf)
    msp_in = doc_in.modelspace()

    line_set = set()
    text_geo_to_entity = {}
    mtext_geo_to_entity = {}

    # Process LINE entities
    for line in msp_in.query("LINE"):
        start = qpt3(line.dxf.start)
        end = qpt3(line.dxf.end)
        key = tuple(sorted([start, end]))
        line_set.add(key)

    # Process LWPOLYLINE entities
    for pl in msp_in.query("LWPOLYLINE"):
        pts = list(pl.get_points("xy"))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                a = (qr(pts[i][0]), qr(pts[i][1]), 0.0)
                b = (qr(pts[i + 1][0]), qr(pts[i + 1][1]), 0.0)
                key = tuple(sorted([a, b]))
                line_set.add(key)
            if pl.closed and len(pts) > 2:
                a = (qr(pts[-1][0]), qr(pts[-1][1]), 0.0)
                b = (qr(pts[0][0]), qr(pts[0][1]), 0.0)
                key = tuple(sorted([a, b]))
                line_set.add(key)

    # Process TEXT entities
    for tx in msp_in.query("TEXT"):
        ins = qpt3(tx.dxf.insert)
        h = qr(tx.dxf.height) if tx.dxf.hasattr("height") else 0.0
        rot = qr(tx.dxf.rotation) if tx.dxf.hasattr("rotation") else 0.0
        key = ("TEXT", ins, h, rot)
        if key not in text_geo_to_entity:
            text_geo_to_entity[key] = tx

    # Process MTEXT entities
    for mt in msp_in.query("MTEXT"):
        ins = qpt3(mt.dxf.insert)
        h = qr(mt.dxf.char_height) if mt.dxf.hasattr("char_height") else 0.0
        w = qr(mt.dxf.width) if mt.dxf.hasattr("width") else 0.0
        rot = qr(mt.dxf.rotation) if mt.dxf.hasattr("rotation") else 0.0
        key = ("MTEXT", ins, h, w, rot)
        if key not in mtext_geo_to_entity:
            mtext_geo_to_entity[key] = mt

    # Create new DXF with deduplicated entities
    doc_out = ezdxf.new(setup=True)
    msp_out = doc_out.modelspace()

    for start, end in line_set:
        msp_out.add_line(start, end)

    for key, tx in text_geo_to_entity.items():
        _, ins, h, rot = key
        new_tx = msp_out.add_text(tx.dxf.text or "", dxfattribs={
            "height": h,
            "rotation": rot,
            "layer": tx.dxf.layer if tx.dxf.hasattr("layer") else "0",
        })
        new_tx.set_pos(ins)

    for key, mt in mtext_geo_to_entity.items():
        _, ins, h, w, rot = key
        new_mt = msp_out.add_mtext(mt.text or "", dxfattribs={
            "char_height": h,
            "width": w,
            "rotation": rot,
            "layer": mt.dxf.layer if mt.dxf.hasattr("layer") else "0",
        })
        new_mt.set_location(ins)

    doc_out.saveas(out_path)
    return out_path



def section_lines_to_shp(input_dxf, related_file_folder, work_dir, output_name, crs):
    """Convert section lines DXF to shapefile with start/end text labels."""
    os.makedirs(work_dir, exist_ok=True)

    lines_dxf = os.path.join(related_file_folder, "LINES.dxf")
    texts_dxf = os.path.join(related_file_folder, "TEXTS.dxf")
    output_shp = os.path.join(work_dir, output_name)

    if not os.path.exists(input_dxf):
        raise FileNotFoundError(f"Input DXF not found: {input_dxf}")

    # Split DXF into lines and texts
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

    line_entities = []
    text_entities = []
    
    for e in msp:
        etype = e.dxftype()
        if etype in ["LINE", "LWPOLYLINE"]:
            line_entities.append(e)
        elif etype in ["TEXT", "MTEXT"]:
            text_entities.append(e)

    if not line_entities:
        raise ValueError("No line entities found in input DXF.")
    if not text_entities:
        raise ValueError("No text entities found in input DXF.")

    # Save lines DXF
    doc_lines = ezdxf.new(setup=True)
    msp_lines = doc_lines.modelspace()
    for e in line_entities:
        try:
            msp_lines.add_foreign_entity(e)
        except Exception as ex:
            print(f"Skipped line entity: {ex}")
    doc_lines.saveas(lines_dxf)

    # Save texts DXF
    doc_texts = ezdxf.new(setup=True)
    msp_texts = doc_texts.modelspace()
    for e in text_entities:
        try:
            msp_texts.add_foreign_entity(e)
        except Exception as ex:
            print(f"Skipped text entity: {ex}")
    doc_texts.saveas(texts_dxf)

    # Convert lines to GeoDataFrame
    doc_lines_r = ezdxf.readfile(lines_dxf)
    msp_lines_r = doc_lines_r.modelspace()

    lines = []
    for line in msp_lines_r.query("LINE"):
        start = (float(line.dxf.start.x), float(line.dxf.start.y))
        end = (float(line.dxf.end.x), float(line.dxf.end.y))
        lines.append({
            "geometry": LineString([start, end]),
            "start": start,
            "end": end
        })

    for pline in msp_lines_r.query("LWPOLYLINE"):
        pts = list(pline.get_points("xy"))
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                start = (float(pts[i][0]), float(pts[i][1]))
                end = (float(pts[i+1][0]), float(pts[i+1][1]))
                lines.append({
                    "geometry": LineString([start, end]),
                    "start": start,
                    "end": end
                })
            if pline.closed:
                start = (float(pts[-1][0]), float(pts[-1][1]))
                end = (float(pts[0][0]), float(pts[0][1]))
                lines.append({
                    "geometry": LineString([start, end]),
                    "start": start,
                    "end": end
                })

    if not lines:
        raise ValueError("No line geometries extracted from lines DXF.")

    gdf_lines = gpd.GeoDataFrame(lines, crs=crs)
    gdf_lines = gdf_lines.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

    # Convert texts to GeoDataFrame
    doc_texts_r = ezdxf.readfile(texts_dxf)
    msp_texts_r = doc_texts_r.modelspace()

    texts = []
    for t in msp_texts_r.query("TEXT MTEXT"):
        content = ""
        if t.dxftype() == "TEXT":
            content = getattr(t.dxf, "text", "") or ""
        else:
            content = getattr(t, "text", "") or ""
        content = content.strip()
        
        insert_x = float(t.dxf.insert.x)
        insert_y = float(t.dxf.insert.y)
        texts.append({
            "geometry": Point((insert_x, insert_y)),
            "text": content
        })

    if not texts:
        raise ValueError("No text geometries extracted from texts DXF.")

    gdf_texts = gpd.GeoDataFrame(texts, crs=crs)
    gdf_texts = gdf_texts.drop_duplicates(subset=["geometry", "text"]).reset_index(drop=True)

    # Find nearest text for each line endpoint
    text_coords = [(p.x, p.y) for p in gdf_texts.geometry]
    tree = cKDTree(text_coords)

    def nearest_text_for_point(pt):
        dist, idx = tree.query([pt.x, pt.y], k=1)
        return gdf_texts.iloc[int(idx)]["text"]

    start_labels = []
    end_labels = []
    
    for _, row in gdf_lines.iterrows():
        start_point = Point(row["start"])
        end_point = Point(row["end"])
        start_labels.append(nearest_text_for_point(start_point))
        end_labels.append(nearest_text_for_point(end_point))

    gdf_lines["start_text"] = start_labels
    gdf_lines["end_text"] = end_labels

    gdf_lines.to_file(output_shp)
    print(f"Final line shapefile with start/end texts saved: {output_shp}")

    return output_shp