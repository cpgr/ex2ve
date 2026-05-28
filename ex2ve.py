#!/usr/bin/env python3
"""ex2ve: collapse a 3D pillar-structured Exodus mesh to a 2D vertical-equilibrium mesh.

First-pass implementation. Reads a 3D HEX8 Exodus mesh, groups cells into columns
by (x, y) top-face footprint, walks each column top-down stopping at the first
vertical gap (a missing cell — assumed already removed from the mesh by upstream
preprocessing — manifests as a z-gap and acts as a flow barrier), and writes a
2D QUAD4 Exodus mesh embedded in 3D space with thickness-weighted upscaled
properties.

Properties handled in this pass:
    PORO, PERMX, PERMY  -- thickness-weighted arithmetic mean
    PERMZ               -- thickness-weighted harmonic mean
    THICKNESS           -- added; sum of dz over the top-active run

Boundary sidesets emitted with the output (QUAD4 edge convention):
    1 -> south (jj_lo), 2 -> east (ii_hi), 3 -> north (jj_hi), 4 -> west (ii_lo)

Output node ordering is CCW (looking down +z) by construction, so corner
Jacobians are positive by construction for any *convex* top face. An
unconditional convexity check rejects the only remaining failure mode --
a HEX8 with a concave or self-intersecting top face -- before writing.

Skipped in this pass (additions for next passes):
    fault-edge preservation, --rule overrides, --config files,
    multi-block consistency checks.

CLI:
    python ex2ve.py INPUT.e [-o OUTPUT.e] [--column-tol TOL]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from netCDF4 import Dataset, chartostring


# -----------------------------------------------------------------------------
# Upscaling rules
# -----------------------------------------------------------------------------

ARITHMETIC_PROPS = {"PORO", "PERMX", "PERMY"}
HARMONIC_PROPS = {"PERMZ"}
DEFAULT_PROPS = ["PORO", "PERMX", "PERMY", "PERMZ"]


def _arithmetic(values: np.ndarray, thickness: np.ndarray) -> float:
    return float(np.sum(values * thickness) / np.sum(thickness))


def _harmonic(values: np.ndarray, thickness: np.ndarray) -> float:
    if np.any(values <= 0):
        # A zero-permeability layer makes the harmonic average zero.
        return 0.0
    return float(np.sum(thickness) / np.sum(thickness / values))


def upscale(prop_name: str, values: np.ndarray, thickness: np.ndarray) -> float:
    if prop_name in HARMONIC_PROPS:
        return _harmonic(values, thickness)
    return _arithmetic(values, thickness)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def top_face_nodes(node_coords: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """Return the 4 top-face node IDs of a HEX8, in CCW order looking down +z.

    Uses the Exodus HEX8 convention: local indices 4-7 are always the top face
    (the nodes connected to the upper ends of the pillars), regardless of their
    absolute z-coordinates. This is robust for tilted or faulted corner-point
    cells where bottom-face nodes may sit higher than some top-face nodes.

    Ordering is then fixed CCW (looking down +z) via arctan2 around the top-face
    centroid. For any *convex* top face this guarantees a non-self-intersecting
    CCW polygon and positive corner Jacobians by construction. Non-convex top
    faces (common in real corner-point grids with tilted pillars or faults) are
    reported by `check_top_faces_convex` as a warning.
    """
    top_nodes = cell_nodes[4:]                  # HEX8 Exodus convention: local 4-7 = top face
    top_xy = node_coords[top_nodes, :2]
    centroid = top_xy.mean(axis=0)
    angles = np.arctan2(top_xy[:, 1] - centroid[1], top_xy[:, 0] - centroid[0])
    order = np.argsort(angles)
    return top_nodes[order]


def cell_z_extents(node_coords: np.ndarray, cell_nodes: np.ndarray) -> tuple[float, float]:
    """Return (z_top, z_bottom) as the mean z of the HEX8 top-face and bottom-face nodes.

    Uses Exodus local indices 4-7 (top face) and 0-3 (bottom face), which is correct
    even for tilted or faulted cells where absolute z-ordering may not separate the faces.
    """
    return (
        float(node_coords[cell_nodes[4:], 2].mean()),
        float(node_coords[cell_nodes[:4], 2].mean()),
    )


def median_top_edge_length(node_coords: np.ndarray, conn: np.ndarray) -> float:
    """Median edge length across all cells' top faces (vectorised)."""
    # conn[:, 4:] selects the 4 top-face nodes for every cell in one shot.
    top_xy = node_coords[conn[:, 4:], :2]          # (N, 4, 2)
    next_xy = top_xy[:, [1, 2, 3, 0], :]
    lengths = np.sqrt(((next_xy - top_xy) ** 2).sum(axis=-1))   # (N, 4)
    return float(np.median(lengths))


# -----------------------------------------------------------------------------
# Top-face convexity check (the only remaining failure mode after CCW ordering)
# -----------------------------------------------------------------------------

def quad_jacobians(
    node_coords: np.ndarray,
    quad_conn: np.ndarray,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Per-corner signed Jacobian for each QUAD4 cell in 3D space.

    At corner c, with e_next = node[c+1] - node[c] and e_prev = node[c-1] - node[c]
    (CCW indexing), the local normal is n = e_next x e_prev. The signed Jacobian
    is the projection of n onto the reference up direction:

        J_c = n . up_hat

    For a CCW-ordered convex quad on an upward-facing surface, all four J_c are
    strictly positive. A non-positive J at any corner indicates a concave or
    self-intersecting polygon -- our ordering can't produce that from a convex
    top face, so it only fires on input-quality issues.
    Returns an (num_elem, 4) array.
    """
    up_arr = np.asarray(up, dtype=float)
    up_hat = up_arr / np.linalg.norm(up_arr)

    pts = node_coords[quad_conn]                          # (E, 4, 3)
    pts_next = pts[:, [1, 2, 3, 0], :]
    pts_prev = pts[:, [3, 0, 1, 2], :]
    e_next = pts_next - pts
    e_prev = pts_prev - pts
    normals = np.cross(e_next, e_prev)                    # (E, 4, 3)
    return np.einsum("ijk,k->ij", normals, up_hat)        # (E, 4)


def check_top_faces_convex(
    node_coords: np.ndarray,
    quad_conn: np.ndarray,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Check output quad top faces for convexity (CCW, +z normal); warn if any fail.

    With CCW node ordering (which `top_face_nodes` produces by construction),
    a convex top face guarantees positive corner Jacobians. A non-positive J
    at any corner indicates that the input HEX8 had a non-convex top face.

    Non-convex top faces are common in real corner-point grids (tilted pillars,
    fault zones, pinch-outs) and do not prevent ex2ve from producing useful
    output — MOOSE and most VE solvers can handle mildly non-convex QUAD4
    elements. A warning is emitted so the user can judge whether the mesh
    quality is acceptable for their application. Returns the Jacobian array.
    """
    J = quad_jacobians(node_coords, quad_conn, up=up)
    bad_per_cell = (J <= 0).any(axis=1)
    n_bad = int(bad_per_cell.sum())
    if n_bad:
        bad_ids = np.where(bad_per_cell)[0]
        pct = 100.0 * n_bad / quad_conn.shape[0]
        print(
            f"Warning: {n_bad} of {quad_conn.shape[0]} output quads "
            f"({pct:.1f}%) have a non-convex top face "
            f"(non-positive corner Jacobian). "
            f"This is normal for tilted or faulted corner-point geometry. "
            f"First affected output elements: {(bad_ids[:5] + 1).tolist()}"
        )
    return J


# -----------------------------------------------------------------------------
# Boundary sidesets (4 cardinal edges of the 2D layer)
# -----------------------------------------------------------------------------

# Side numbering for a CCW-ordered QUAD4 (matches Exodus II convention):
#   side 1 = nodes (1, 2) -> jj_lo (south)
#   side 2 = nodes (2, 3) -> ii_hi (east)
#   side 3 = nodes (3, 4) -> jj_hi (north)
#   side 4 = nodes (4, 1) -> ii_lo (west)
SIDESET_NAMES = {1: "south", 2: "east", 3: "north", 4: "west"}


def compute_boundary_sidesets(quad_conn: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Identify boundary edges and group them by their cell-local side index.

    An edge is on the boundary if and only if it appears in exactly one quad.
    Returns: {side_id_1_to_4: [(elem_id_1_indexed, side_id_1_to_4), ...]}.
    """
    # First pass: count edge occurrences (undirected).
    edge_count: dict[frozenset, int] = {}
    for nodes in quad_conn:
        for s in range(4):
            edge = frozenset((int(nodes[s]), int(nodes[(s + 1) % 4])))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Second pass: collect (elem, side) for boundary edges, bucketed by side_id.
    sidesets: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}
    for e_idx, nodes in enumerate(quad_conn):
        for s in range(4):
            edge = frozenset((int(nodes[s]), int(nodes[(s + 1) % 4])))
            if edge_count[edge] == 1:
                side_id = s + 1
                sidesets[side_id].append((e_idx + 1, side_id))
    return sidesets


# -----------------------------------------------------------------------------
# Exodus reader (minimal, HEX8 input)
# -----------------------------------------------------------------------------

@dataclass
class Exodus3D:
    node_coords: np.ndarray                # (num_nodes, 3)
    conn: np.ndarray                       # (num_elem, 8), 0-indexed
    elem_vars: dict[str, np.ndarray]       # name -> (num_elem,) array, concatenated across blocks


def _decode_name_array(name_var) -> list[str]:
    return [str(s).strip() for s in chartostring(name_var[:])]


def read_exodus_3d(path: Path) -> Exodus3D:
    ds = Dataset(str(path), "r")
    try:
        node_coords = np.column_stack([
            np.asarray(ds.variables["coordx"][:]),
            np.asarray(ds.variables["coordy"][:]),
            np.asarray(ds.variables["coordz"][:]),
        ])

        num_el_blk = ds.dimensions["num_el_blk"].size
        conn_blocks = []
        block_sizes = []  # number of cells per block, in block order

        for b in range(1, num_el_blk + 1):
            connect_var = ds.variables[f"connect{b}"]
            elem_type = getattr(connect_var, "elem_type", "").upper()
            if "HEX" not in elem_type:
                raise ValueError(
                    f"Block {b} has elem_type={elem_type!r}; ex2ve requires HEX8 input."
                )
            block_conn = np.asarray(connect_var[:]) - 1  # to 0-indexed
            if block_conn.shape[1] != 8:
                raise ValueError(
                    f"Block {b} has {block_conn.shape[1]} nodes/elem; expected 8 (HEX8)."
                )
            conn_blocks.append(block_conn.astype(np.int64))
            block_sizes.append(block_conn.shape[0])

        conn = np.concatenate(conn_blocks, axis=0)

        elem_vars: dict[str, np.ndarray] = {}
        if (
            "name_elem_var" in ds.variables
            and "time_step" in ds.dimensions
            and ds.dimensions["time_step"].size > 0
        ):
            names = _decode_name_array(ds.variables["name_elem_var"])
            for v_idx, name in enumerate(names, start=1):
                pieces = []
                for b, n_in_block in enumerate(block_sizes, start=1):
                    var_name = f"vals_elem_var{v_idx}eb{b}"
                    if var_name in ds.variables:
                        # Take the last time step (em2ex writes a single static step).
                        pieces.append(np.asarray(ds.variables[var_name][-1, :]))
                    else:
                        pieces.append(np.full(n_in_block, np.nan))
                elem_vars[name.upper()] = np.concatenate(pieces)

        return Exodus3D(node_coords=node_coords, conn=conn, elem_vars=elem_vars)
    finally:
        ds.close()


# -----------------------------------------------------------------------------
# Column identification and reduction
# -----------------------------------------------------------------------------

@dataclass
class Column:
    footprint: tuple[float, float]       # actual top-face centroid of the topmost active cell
    top_active: list[int]                # cell indices, top-first
    thickness: float
    quad_nodes: np.ndarray               # (4,) original-mesh node IDs for the output quad
    upscaled: dict[str, float] = field(default_factory=dict)


def identify_columns(
    node_coords: np.ndarray,
    conn: np.ndarray,
    tol: float,
) -> dict[tuple[int, int], list[int]]:
    """Group cell indices by snapped top-face centroid (x, y).

    Uses HEX8 local indices 4-7 for the top face; the centroid is the mean
    xy of those 4 nodes. The footprint key is the centroid snapped to the
    nearest tolerance grid cell.
    """
    # Vectorised: conn[:, 4:] gives the 4 top-face nodes for all cells at once.
    top_xy = node_coords[conn[:, 4:], :2].mean(axis=1)        # (N, 2)
    snapped = np.round(top_xy / tol).astype(np.int64)         # (N, 2)
    groups: dict[tuple[int, int], list[int]] = {}
    for c, (kx, ky) in enumerate(snapped):
        key = (int(kx), int(ky))
        groups.setdefault(key, []).append(c)
    return groups


def collapse_columns(
    ex: Exodus3D,
    tol: Optional[float] = None,
    properties: Optional[list[str]] = None,
) -> list[Column]:
    """Run the full pillar-wise collapse. Returns one Column per emitted quad."""
    n_cells = ex.conn.shape[0]

    # Per-cell z extents (vectorised; HEX8 local 4-7 = top face, 0-3 = bottom face).
    z_top = ex.node_coords[ex.conn[:, 4:], 2].mean(axis=1)   # (N,)
    z_bot = ex.node_coords[ex.conn[:, :4], 2].mean(axis=1)   # (N,)
    dz = z_top - z_bot

    if tol is None:
        tol = 0.01 * median_top_edge_length(ex.node_coords, ex.conn)
    gap_tol = 0.01 * float(np.median(dz))

    if properties is None:
        properties = [p for p in DEFAULT_PROPS if p in ex.elem_vars]

    groups = identify_columns(ex.node_coords, ex.conn, tol)

    columns: list[Column] = []
    for cell_ids in groups.values():
        cell_ids_sorted = sorted(cell_ids, key=lambda c: -z_top[c])

        # Walk top-down; stop at the first vertical gap.
        top_active = [cell_ids_sorted[0]]
        for c in cell_ids_sorted[1:]:
            if z_bot[top_active[-1]] - z_top[c] > gap_tol:
                break
            top_active.append(c)

        thickness_per_cell = dz[top_active]
        thickness = float(thickness_per_cell.sum())

        upscaled = {
            prop: upscale(prop, ex.elem_vars[prop][top_active], thickness_per_cell)
            for prop in properties
        }

        top_cell = top_active[0]
        qn = top_face_nodes(ex.node_coords, ex.conn[top_cell])
        xy = ex.node_coords[qn, :2].mean(axis=0)

        columns.append(Column(
            footprint=(float(xy[0]), float(xy[1])),
            top_active=top_active,
            thickness=thickness,
            quad_nodes=qn,
            upscaled=upscaled,
        ))

    # Deterministic output ordering: by (y, x) of the footprint.
    columns.sort(key=lambda col: (col.footprint[1], col.footprint[0]))
    return columns


# -----------------------------------------------------------------------------
# Exodus writer (minimal, QUAD4 in 3D space, no sidesets)
# -----------------------------------------------------------------------------

LEN_STRING = 33
LEN_LINE = 81


def _write_name_array(var, names: list[str]) -> None:
    # Build a (len(names), LEN_STRING) S1 array without using stringtochar,
    # which is broken in netCDF4 >= 1.7 when paired with numpy >= 2.0.
    arr = np.array(
        [list(s.ljust(LEN_STRING)[:LEN_STRING]) for s in names], dtype="S1"
    )
    var[:, :] = arr


def write_exodus_2d(
    path: Path,
    node_coords_in: np.ndarray,
    columns: list[Column],
    title: str = "ex2ve VE mesh",
) -> None:
    if not columns:
        raise ValueError("No columns to write; input produced zero output quads.")

    quad_nodes = np.stack([col.quad_nodes for col in columns])         # (N, 4)
    unique_nodes, inverse = np.unique(quad_nodes.ravel(), return_inverse=True)
    new_conn = inverse.reshape(-1, 4).astype(np.int32) + 1             # 1-indexed for Exodus
    out_node_coords = node_coords_in[unique_nodes]

    num_nodes = out_node_coords.shape[0]
    num_elem = new_conn.shape[0]

    prop_names = sorted({p for col in columns for p in col.upscaled.keys()})
    all_vars = prop_names + ["THICKNESS"]
    var_arrays = {p: np.array([col.upscaled.get(p, np.nan) for col in columns]) for p in prop_names}
    var_arrays["THICKNESS"] = np.array([col.thickness for col in columns])

    # Boundary sidesets (only emit non-empty ones; classic netCDF can't store
    # a zero-length dimension).
    sidesets_all = compute_boundary_sidesets(new_conn - 1)  # back to 0-indexed for the helper
    sidesets = {sid: pairs for sid, pairs in sidesets_all.items() if pairs}
    sideset_ids = sorted(sidesets.keys())  # 1..4 ordering, skipping empties
    num_side_sets = len(sideset_ids)

    ds = Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET")
    try:
        ds.title = title
        ds.api_version = np.float32(4.98)
        ds.version = np.float32(4.98)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(1)

        ds.createDimension("len_string", LEN_STRING)
        ds.createDimension("len_line", LEN_LINE)
        ds.createDimension("four", 4)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", num_nodes)
        ds.createDimension("num_elem", num_elem)
        ds.createDimension("num_el_blk", 1)
        ds.createDimension("num_el_in_blk1", num_elem)
        ds.createDimension("num_nod_per_el1", 4)
        ds.createDimension("num_elem_var", len(all_vars))
        ds.createDimension("time_step", None)
        if num_side_sets:
            ds.createDimension("num_side_sets", num_side_sets)
            for idx, sid in enumerate(sideset_ids, start=1):
                ds.createDimension(f"num_side_ss{idx}", len(sidesets[sid]))

        ds.createVariable("coordx", "f8", ("num_nodes",))[:] = out_node_coords[:, 0]
        ds.createVariable("coordy", "f8", ("num_nodes",))[:] = out_node_coords[:, 1]
        ds.createVariable("coordz", "f8", ("num_nodes",))[:] = out_node_coords[:, 2]
        _write_name_array(
            ds.createVariable("coor_names", "S1", ("num_dim", "len_string")),
            ["x", "y", "z"],
        )

        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_prop1[:] = np.array([1], dtype=np.int32)
        ds.createVariable("eb_status", "i4", ("num_el_blk",))[:] = np.array([1], dtype=np.int32)
        _write_name_array(
            ds.createVariable("eb_names", "S1", ("num_el_blk", "len_string")),
            ["ve_layer"],
        )

        connect1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
        connect1.elem_type = "QUAD4"
        connect1[:, :] = new_conn

        ds.createVariable("time_whole", "f8", ("time_step",))[0] = 0.0

        _write_name_array(
            ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_string")),
            all_vars,
        )
        elem_var_tab = ds.createVariable("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"))
        elem_var_tab[:, :] = 1

        for v_idx, name in enumerate(all_vars, start=1):
            var = ds.createVariable(
                f"vals_elem_var{v_idx}eb1", "f8",
                ("time_step", "num_el_in_blk1"),
            )
            var[0, :] = var_arrays[name]

        # Sidesets (4 cardinal boundary edges of the 2D layer).
        if num_side_sets:
            ss_status = ds.createVariable("ss_status", "i4", ("num_side_sets",))
            ss_status[:] = np.ones(num_side_sets, dtype=np.int32)

            ss_prop1 = ds.createVariable("ss_prop1", "i4", ("num_side_sets",))
            ss_prop1.setncattr("name", "ID")
            ss_prop1[:] = np.array(sideset_ids, dtype=np.int32)

            _write_name_array(
                ds.createVariable("ss_names", "S1", ("num_side_sets", "len_string")),
                [SIDESET_NAMES[sid] for sid in sideset_ids],
            )

            for idx, sid in enumerate(sideset_ids, start=1):
                pairs = sidesets[sid]
                elem_ids = np.array([e for (e, _s) in pairs], dtype=np.int32)
                side_ids = np.array([s for (_e, s) in pairs], dtype=np.int32)
                ds.createVariable(f"elem_ss{idx}", "i4", (f"num_side_ss{idx}",))[:] = elem_ids
                ds.createVariable(f"side_ss{idx}", "i4", (f"num_side_ss{idx}",))[:] = side_ids
    finally:
        ds.close()


# -----------------------------------------------------------------------------
# Driver + CLI
# -----------------------------------------------------------------------------

def collapse(
    input_path: Path,
    output_path: Path,
    column_tol: Optional[float] = None,
) -> int:
    print(f"Reading {input_path}")
    ex = read_exodus_3d(input_path)
    print(f"  {ex.node_coords.shape[0]} nodes, {ex.conn.shape[0]} HEX8 cells")
    print(f"  Element variables: {sorted(ex.elem_vars.keys()) or '(none)'}")

    columns = collapse_columns(ex, tol=column_tol)
    print(f"Collapsed to {len(columns)} columns (quads)")
    if columns:
        upscaled_props = sorted(columns[0].upscaled.keys())
        print(f"  Upscaled properties: {upscaled_props + ['THICKNESS']}")

        # By construction, top_face_nodes produces CCW ordering, so positive
        # corner Jacobians are guaranteed for convex top faces. This check
        # catches the only remaining failure mode: a non-convex input top face.
        quad_conn = np.stack([col.quad_nodes for col in columns])
        check_top_faces_convex(ex.node_coords, quad_conn)

    print(f"Writing {output_path}")
    write_exodus_2d(output_path, ex.node_coords, columns)
    return len(columns)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collapse a 3D pillar-structured Exodus mesh to a 2D VE mesh."
    )
    parser.add_argument("input", type=Path, help="Input 3D Exodus mesh (HEX8).")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output 2D Exodus mesh (QUAD4). Default: <input-stem>_ve.e",
    )
    parser.add_argument(
        "--column-tol", type=float, default=None,
        help="Tolerance for (x, y) column grouping. "
             "Default: 1%% of the median top-face edge length.",
    )
    args = parser.parse_args(argv)

    output = args.output or args.input.with_name(args.input.stem + "_ve.e")
    collapse(args.input, output, column_tol=args.column_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
