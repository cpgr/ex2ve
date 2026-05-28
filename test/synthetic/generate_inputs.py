#!/usr/bin/env python
"""Generate synthetic 3D HEX8 Exodus fixtures for the ex2ve test harness.

Run this once to (re)create the input .e files referenced by ``tests``:

    python test/synthetic/generate_inputs.py

Three fixtures are produced, each at the same path as this script:

    simple_uniform.e   2x2x3 cells, all active, uniform properties per layer
    barrier.e          2x2x4 cells with the middle layer removed (flow barrier)
    heterogeneous.e    2x2x2 cells, each (i,j) column gets distinct properties

The analytical expected outputs are documented in the ``tests`` YAML next door.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset, stringtochar


LEN_STRING = 33
LEN_LINE = 81


def _write_name_array(var, names: list[str]) -> None:
    arr = np.array(names, dtype=f"S{LEN_STRING}")
    var[:, :] = stringtochar(arr)


def write_block_exodus(
    path: Path,
    node_coords: np.ndarray,         # (num_nodes, 3)
    conn_0idx: np.ndarray,           # (num_elem, nodes_per_elem) 0-indexed
    elem_vars: dict[str, np.ndarray],
    elem_type: str = "HEX8",
    title: str = "ex2ve test fixture",
) -> None:
    """Write a minimal single-block Exodus file with the given elem_type."""
    num_nodes = node_coords.shape[0]
    num_elem = conn_0idx.shape[0]
    nodes_per_elem = conn_0idx.shape[1]
    var_names = list(elem_vars.keys())

    with Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as ds:
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
        ds.createDimension("num_nod_per_el1", nodes_per_elem)
        if var_names:
            ds.createDimension("num_elem_var", len(var_names))
        ds.createDimension("time_step", None)

        ds.createVariable("coordx", "f8", ("num_nodes",))[:] = node_coords[:, 0]
        ds.createVariable("coordy", "f8", ("num_nodes",))[:] = node_coords[:, 1]
        ds.createVariable("coordz", "f8", ("num_nodes",))[:] = node_coords[:, 2]
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
            ["reservoir"],
        )

        connect1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
        connect1.elem_type = elem_type
        connect1[:, :] = conn_0idx.astype(np.int32) + 1

        ds.createVariable("time_whole", "f8", ("time_step",))[0] = 0.0

        if var_names:
            _write_name_array(
                ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_string")),
                var_names,
            )
            ds.createVariable("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"))[:, :] = 1
            for v_idx, name in enumerate(var_names, start=1):
                var = ds.createVariable(
                    f"vals_elem_var{v_idx}eb1", "f8",
                    ("time_step", "num_el_in_blk1"),
                )
                var[0, :] = elem_vars[name]


def cartesian_pillar_mesh(
    nx: int, ny: int, nz: int,
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a structured Cartesian HEX8 mesh.

    Returns (node_coords[(nx+1)*(ny+1)*(nz+1), 3], conn[nx*ny*nz, 8]).
    Cell ordering: i fastest, j next, k slowest.
    Node ordering: same (i, j, k) raster.
    Per-cell node ordering: Exodus HEX8 standard
        bottom (z low):  (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        top    (z high): same xy CCW, +1 in k
    """
    ox, oy, oz = origin
    xs = ox + np.arange(nx + 1) * dx
    ys = oy + np.arange(ny + 1) * dy
    zs = oz + np.arange(nz + 1) * dz

    def nid(i: int, j: int, k: int) -> int:
        return (k * (ny + 1) + j) * (nx + 1) + i

    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = (xs[i], ys[j], zs[k])

    conn = np.zeros((nx * ny * nz, 8), dtype=np.int64)
    e = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn[e] = [
                    nid(i,     j,     k),
                    nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),
                    nid(i,     j + 1, k),
                    nid(i,     j,     k + 1),
                    nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i,     j + 1, k + 1),
                ]
                e += 1
    return nodes, conn


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

def make_simple_uniform(out_dir: Path) -> None:
    """2x2x3 cells, dz=1. Uniform per-layer properties; no barriers.

    Layer values, k=0 (bottom)..k=2 (top):
        PORO  = 0.1, 0.3, 0.2
        PERMX = 50,  200, 100   (PERMY identical)
        PERMZ = 5,   20,  10
    """
    nodes, conn = cartesian_pillar_mesh(2, 2, 3, dz=1.0)
    n_cells = conn.shape[0]  # 12

    poro = np.zeros(n_cells)
    poro[0:4] = 0.1    # k=0
    poro[4:8] = 0.3    # k=1
    poro[8:12] = 0.2   # k=2 (top)

    permx = np.zeros(n_cells)
    permx[0:4] = 50.0
    permx[4:8] = 200.0
    permx[8:12] = 100.0

    permz = np.zeros(n_cells)
    permz[0:4] = 5.0
    permz[4:8] = 20.0
    permz[8:12] = 10.0

    elem_vars = {
        "PORO": poro,
        "PERMX": permx,
        "PERMY": permx.copy(),
        "PERMZ": permz,
    }
    write_block_exodus(out_dir / "simple_uniform.e", nodes, conn, elem_vars)


def make_barrier(out_dir: Path) -> None:
    """2x2x4 cells, dz=1, with the k=1 layer omitted from the mesh.

    Active layers in the mesh: k=0 (bottom), k=2, k=3 (top). The gap at z=[1, 2]
    is a flow barrier; ex2ve should keep only the k=3 + k=2 active run.
    """
    nodes, conn = cartesian_pillar_mesh(2, 2, 4, dz=1.0)
    keep = np.ones(conn.shape[0], dtype=bool)
    keep[4:8] = False     # drop k=1 cells (4..7)
    conn = conn[keep]

    # After dropping: cells 0..3 are k=0, 4..7 are k=2, 8..11 are k=3 (top).
    poro = np.zeros(conn.shape[0])
    poro[0:4] = 0.1
    poro[4:8] = 0.3
    poro[8:12] = 0.2

    permx = np.zeros(conn.shape[0])
    permx[0:4] = 50.0
    permx[4:8] = 200.0
    permx[8:12] = 100.0

    permz = np.zeros(conn.shape[0])
    permz[0:4] = 5.0
    permz[4:8] = 20.0
    permz[8:12] = 10.0

    elem_vars = {
        "PORO": poro,
        "PERMX": permx,
        "PERMY": permx.copy(),
        "PERMZ": permz,
    }
    write_block_exodus(out_dir / "barrier.e", nodes, conn, elem_vars)


def make_heterogeneous(out_dir: Path) -> None:
    """2x2x2 cells, dz=1. Each (i,j) column gets distinct per-layer PORO.

    Bottom (k=0) PORO by (i,j):  (0,0)=0.10, (1,0)=0.15, (0,1)=0.20, (1,1)=0.05
    Top    (k=1) PORO by (i,j):  (0,0)=0.20, (1,0)=0.25, (0,1)=0.30, (1,1)=0.15

    Column means (equal-thickness arithmetic):
        (0,0) = 0.15, (1,0) = 0.20, (0,1) = 0.25, (1,1) = 0.10
    """
    nodes, conn = cartesian_pillar_mesh(2, 2, 2, dz=1.0)
    poro = np.array([
        # k=0 (bottom), then k=1 (top); within each layer (i,j) raster (i fastest)
        0.10, 0.15, 0.20, 0.05,
        0.20, 0.25, 0.30, 0.15,
    ])
    elem_vars = {"PORO": poro}
    write_block_exodus(out_dir / "heterogeneous.e", nodes, conn, elem_vars)


def make_single_cell(out_dir: Path) -> None:
    """1x1x1 mesh. Minimum viable input: one column, one cell, no averaging.

    Tests that the algorithm handles the degenerate single-cell case
    (no division-by-zero in weighted means, single-cell harmonic = identity).
    Uses dz=2 to make THICKNESS distinguishable from the cell count.
    """
    nodes, conn = cartesian_pillar_mesh(1, 1, 1, dz=2.0)
    elem_vars = {
        "PORO": np.array([0.30]),
        "PERMX": np.array([75.0]),
        "PERMY": np.array([75.0]),
        "PERMZ": np.array([12.0]),
    }
    write_block_exodus(out_dir / "single_cell.e", nodes, conn, elem_vars)


def make_permz_zero(out_dir: Path) -> None:
    """2x2x3 mesh where PERMZ=0 in the middle layer.

    Tests the harmonic-mean short-circuit: any zero-permeability layer
    drives the column's effective PERMZ to 0. The other properties
    are unaffected (arithmetic mean, normal averaging).
    """
    nodes, conn = cartesian_pillar_mesh(2, 2, 3, dz=1.0)
    n = conn.shape[0]
    poro = np.zeros(n);   poro[0:4] = 0.1;   poro[4:8] = 0.3;   poro[8:12] = 0.2
    permx = np.zeros(n);  permx[0:4] = 50;   permx[4:8] = 200;  permx[8:12] = 100
    permz = np.zeros(n);  permz[0:4] = 5;    permz[4:8] = 0.0;  permz[8:12] = 10
    elem_vars = {
        "PORO": poro,
        "PERMX": permx,
        "PERMY": permx.copy(),
        "PERMZ": permz,
    }
    write_block_exodus(out_dir / "permz_zero.e", nodes, conn, elem_vars)


def make_twisted_quad(out_dir: Path) -> None:
    """A single HEX8 whose top face is a concave (kite-shaped) quad.

    With node (1, 0.5, z_top) lying inside the convex hull of the other three
    top corners, the CCW-by-angle ordering produces a self-overlapping polygon:
    at one corner the local normal flips sign, so the 2D Jacobian check should
    flag a non-positive J. Used to exercise the --strict-jacobians failure path.
    """
    nodes = np.array([
        # bottom face (z=0)
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 2.0, 0.0],
        # top face (z=1) — same (x, y), shifted up
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [1.0, 2.0, 1.0],
    ])
    # HEX8 ordering: bottom 0..3 (CCW from below), top 4..7 (matching pillars).
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    elem_vars = {"PORO": np.array([0.20])}
    write_block_exodus(out_dir / "twisted_quad.e", nodes, conn, elem_vars)


def make_non_hex8(out_dir: Path) -> None:
    """A single tetrahedron written as a TETRA block.

    Tests that ex2ve rejects non-HEX8 input cleanly at read time, regardless
    of cell shape. The mesh has 4 nodes and 1 4-node element with
    elem_type='TETRA'.
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    elem_vars = {"PORO": np.array([0.2])}
    write_block_exodus(out_dir / "non_hex8.e", nodes, conn, elem_vars, elem_type="TETRA")


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    # Core / happy-path fixtures
    make_simple_uniform(out_dir)
    make_barrier(out_dir)
    make_heterogeneous(out_dir)
    # Edge-case fixtures
    make_single_cell(out_dir)
    make_permz_zero(out_dir)
    make_twisted_quad(out_dir)
    make_non_hex8(out_dir)
    print(f"Generated 7 fixtures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
