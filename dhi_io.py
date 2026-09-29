"""I/O for the delivered prospect formats (spec §2, §3.5).

- Petrel "Points with Attributes" ASCII: `#` comment header with `# Column n: <name>`
  lines, then whitespace-delimited numeric rows. Columns are mapped BY NAME from the
  header (never by position) so a re-exported file with extra or reordered attributes
  cannot silently swap fluid and lithology.
- Esri Shapefile container polygon: read with pyshp, turned into a shapely geometry
  (multipart + holes preserved). Point-in-polygon is vectorised via shapely 2.
- YAML manifests (rubric, surveys, per-prospect records).
"""
import re

import numpy as np
import shapefile
import yaml
from shapely.geometry import MultiPolygon, Polygon
from shapely import intersects_xy

_COL_RE = re.compile(r"#\s*Column\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_REQUIRED = ("x", "y", "z", "fluid")


def _parse_header(path):
    """Return {column_name: 0-based index} from the '# Column n: name' lines."""
    names = {}
    col_nums = {}  # track which column number each stripped name was first seen at
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = _COL_RE.match(line)
            if m:
                # header names may carry units in parentheses: "Z (Depth or Two-Way Time)"
                stripped = m.group(2).split("(")[0].strip()
                col_num = int(m.group(1)) - 1
                if stripped in names:
                    # collision: same name (after unit stripping) at different columns
                    raise ValueError(f"{path}: column name '{stripped}' appears at columns {col_nums[stripped] + 1} and {col_num + 1}")
                names[stripped] = col_num
                col_nums[stripped] = col_num
    if not names:
        raise ValueError(f"{path}: no '# Column n: name' header lines found")
    return names


def null_values_of(column_map):
    """The optional `null_values` entry of a column map (Petrel's undefined-value sentinel,
    e.g. -999) as a tuple; a scalar is accepted for a single sentinel."""
    nv = (column_map or {}).get("null_values") or ()
    return tuple(nv) if isinstance(nv, (list, tuple, set)) else (nv,)


def read_petrel_points(path, column_map, null_values=(), return_dropped=False):
    """Mapped columns by name -> {role: float64 array}. Rows where ANY mapped column equals a
    value in `null_values` are dropped (a -999 sentinel would otherwise pass isfinite and
    land in a leg as a wildly bright point); `return_dropped=True` returns (pts, n_dropped)."""
    names = _parse_header(path)
    idx = {}
    for role in ("x", "y", "z", "fluid", "lith"):
        col = column_map.get(role)
        if col is None:
            if role in _REQUIRED:
                raise ValueError(f"column_map has no entry for required role '{role}'")
            continue
        if col not in names:
            raise ValueError(f"{path}: role '{role}' -> column '{col}' not in header {sorted(names)}")
        idx[role] = names[col]
    data = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    keep = np.ones(len(data), dtype=bool)
    if len(null_values):
        cols = data[:, list(idx.values())]
        keep = ~np.isin(cols, np.asarray(null_values, dtype=np.float64)).any(axis=1)
    pts = {role: data[keep, i] for role, i in idx.items()}
    return (pts, int((~keep).sum())) if return_dropped else pts


def read_container_polygon(shp_path):
    """Shapefile rings -> shapely. pyshp gives `parts` offsets; ring orientation decides
    outer (clockwise) vs hole (counter-clockwise) per the shapefile spec."""
    r = shapefile.Reader(shp_path)
    polys = []
    for shp in r.shapes():
        pts = shp.points
        parts = list(shp.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        outers, holes = [], []
        ring_idx_map = {}  # map ring index in rings to its position in outers or holes
        for ring_idx, ring in enumerate(rings):
            if shapefile.signed_area(ring) < 0:
                outers.append(ring)
                ring_idx_map[ring_idx] = ('outer', len(outers) - 1)
            else:
                holes.append(ring)
                ring_idx_map[ring_idx] = ('hole', len(holes) - 1)
        # Check that every hole is contained by at least one outer
        for hole_idx, hole in enumerate(holes):
            hole_contained = any(Polygon(outer).contains(Polygon(hole)) for outer in outers)
            if not hole_contained:
                # Find the original ring index (the hole ring is the ring_idx-th ring overall)
                # We need to find which original ring_idx corresponds to this hole
                original_ring_idx = None
                for orig_idx, (ring_type, idx) in ring_idx_map.items():
                    if ring_type == 'hole' and idx == hole_idx:
                        original_ring_idx = orig_idx
                        break
                raise ValueError(f"{shp_path}: ring {original_ring_idx} is counter-clockwise but lies inside no outer ring")
        for outer in outers:
            poly = Polygon(outer, [h for h in holes if Polygon(outer).contains(Polygon(h))])
            polys.append(poly)
    if not polys:
        raise ValueError(f"{shp_path}: no polygon rings")
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def points_in_polygon(x, y, geom):
    """Boundary counts as inside (`intersects`, vectorised), so a lattice point lying exactly
    on the container edge is not dropped nondeterministically."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return np.asarray(intersects_xy(geom, x, y), dtype=bool)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
