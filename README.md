# ex2ve

**ex2ve** collapses a 3D pillar-structured Exodus mesh into a 2D
[vertical-equilibrium (VE)](https://en.wikipedia.org/wiki/Vertical_equilibrium_model)
mesh suitable for MOOSE.

```
3D Exodus (HEX8)  →  ex2ve  →  2D Exodus (QUAD4)
```

It is a post-processing companion to
[em2ex](https://github.com/idaholab/em2ex) and sits at the end of the
geological-model-to-simulation pipeline:

```
Petrel / grdecl
  └─► em2ex
        └─► 3D Exodus ──┬──► MOOSE  (full 3D simulation)
                        └──► ex2ve
                               └──► 2D Exodus ──► MOOSE  (VE scoping study)
```

The primary use case is **scoping studies**: sweep injection rates, well
locations, or permeability sensitivities at a fraction of the cost of a full
3D simulation, without a round-trip through Petrel to manually upscale.

---

## Contents

- [Why a separate tool?](#why-a-separate-tool)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
  - [Column identification](#1-column-identification)
  - [Barrier-aware reduction](#2-barrier-aware-reduction)
  - [Property upscaling](#3-property-upscaling)
  - [2D mesh construction](#4-2d-mesh-construction)
  - [Boundary sidesets](#5-boundary-sidesets)
- [Output format](#output-format)
- [Running the tests](#running-the-tests)

---

## Why a separate tool?

em2ex has a clean contract: *"translate a geological model into a faithful
3D computational mesh."*  A `--collapse-z` flag would break that contract —
the same script run against the same input would produce fundamentally
different mesh shapes depending on a flag.

The two transforms also live at different abstraction levels:

| | em2ex | ex2ve |
|---|---|---|
| **Domain knowledge** | grdecl (ZCORN, MAPAXES, corner-point pillars) | generic 3D Exodus HEX8 mesh |
| **Input** | `.grdecl` / Eclipse keyword deck | any column-structured Exodus file |
| **Output** | faithful 3D HEX8 Exodus | upscaled 2D QUAD4 Exodus |
| **Naming convention** | — | fits the SEACAS family (`ejoin`, `exodiff`, `epu`, …) |

Keeping them separate means each tool can evolve independently as VE physics
or upscaling rules change, without accumulating flags in a monolith.

---

## Installation

**Python 3.9+** and the following packages are required:

```
numpy>=1.23
netCDF4>=1.6
```

For testing:

```
pytest>=7
pyyaml>=6
```

Install everything in one step:

```bash
pip install -r requirements.txt
```

No build step; `ex2ve.py` is a single self-contained script.

---

## Usage

```
python ex2ve.py INPUT.e [-o OUTPUT.e] [--column-tol TOL]
```

| Argument | Default | Description |
|---|---|---|
| `INPUT.e` | *(required)* | 3D HEX8 Exodus mesh to collapse |
| `-o OUTPUT.e` | `<input-stem>_ve.e` | Output 2D QUAD4 Exodus mesh |
| `--column-tol TOL` | 1 % of median top-face edge length | Tolerance for grouping cells into columns by (x, y) footprint |

### Example

```bash
# Collapse a full-field model for a VE scoping run
python ex2ve.py field_model.e

# Explicit output path and tighter column tolerance
python ex2ve.py field_model.e -o field_model_ve.e --column-tol 0.05
```

Typical console output:

```
Reading field_model.e
  48 000 nodes, 12 000 HEX8 cells
  Element variables: ['PERMX', 'PERMY', 'PERMZ', 'PORO']
Collapsed to 2 000 columns (quads)
  Upscaled properties: ['PERMX', 'PERMY', 'PERMZ', 'PORO', 'THICKNESS']
Writing field_model_ve.e
```

---

## How it works

### 1. Column identification

Cells are grouped into vertical columns using their **top-face (x, y)
centroid**, snapped to a tolerance grid.  Within each column, cells are sorted
by z (top first).

**Default tolerance:** 1 % of the median top-face edge length in the input
mesh.  For a typical 100 m cell spacing this is ~1 m — large enough to absorb
floating-point jitter, small enough to prevent adjacent columns from merging.
Override with `--column-tol` if your mesh has unusually coarse or fine pillar
spacing.

This heuristic works for any pillar-structured mesh regardless of origin
(em2ex, Petrel, or any corner-point converter).  ex2ve does **not** require
`(i, j, k)` metadata in the Exodus file.  Non-pillar inputs (e.g., unstructured
tet meshes, or grids where footprints don't cluster cleanly) are rejected with
a clear error.

Only **HEX8** element blocks are accepted.  A TETRA, WEDGE, or mixed-topology
block is rejected at read time.

### 2. Barrier-aware reduction

A **flow barrier** is a missing cell: a z-gap between two adjacent cells in a
column that is larger than half a nominal cell thickness.  Barriers arise
naturally when upstream preprocessing removes inactive or non-reservoir cells
from the mesh; ex2ve does not read or expect any `ACTNUM`-style activity flag.

For each column, ex2ve walks top-down and collects the **top-active run** — the
contiguous sequence of cells from the topmost cell down to (but not including)
the first gap:

```
k=3  ████  ← top cell       ┐
k=2  ████                   ├── top-active run: upscaled into one QUAD4
k=1  ░░░░  ← gap / barrier  ┘
k=0  ████  ← discarded (separate flow unit below the barrier)
```

Everything below the first barrier is a disconnected lower flow unit and is
discarded.  A column whose **topmost** cell is missing emits no quad at all.

This means columns can have different lengths, partial activity, and irregular
z-spacing without any special treatment — they just produce thinner or thicker
upscaled cells.

### 3. Property upscaling

All upscaling is performed over the **top-active run only**.  Layers below a
barrier never contribute.

| Property | Rule |
|---|---|
| `PORO`, `PERMX`, `PERMY` | Thickness-weighted arithmetic mean |
| `PERMZ` | Thickness-weighted harmonic mean |
| Any other variable present in the input | Thickness-weighted arithmetic mean |
| `THICKNESS` | Sum of layer thicknesses (always added) |

**Arithmetic mean** (for a property *φ* over layers with thickness *h*):

$$\phi_\text{eff} = \frac{\sum_i h_i \phi_i}{\sum_i h_i}$$

**Harmonic mean** (for vertical permeability *k_z*):

$$k_{z,\text{eff}} = \frac{\sum_i h_i}{\sum_i h_i / k_{z,i}}$$

If any layer has *k_z* = 0, the harmonic mean short-circuits to 0 — a
zero-permeability layer makes the whole column a vertical seal.

### 4. 2D mesh construction

The 2D output element for each column reuses the **top face of the topmost
active cell** directly from the 3D mesh.  Nodes are not moved or interpolated;
they keep their original (x, y, z) coordinates, preserving top-surface
topography for MOOSE's VE gravity term.

Nodes are ordered **counter-clockwise** (viewed from above) by sorting around
the top-face centroid using `arctan2`.  This guarantees positive corner
Jacobians for any *convex* top face.  An unconditional convexity check is
applied before writing: if any top face is non-convex (concave kite, twisted
quad), ex2ve raises an error rather than silently emitting an invalid element.

Output quads are ordered by **(y, x)** of the column centroid (y first, then x
within each row), so the output ordering is deterministic and human-readable.

### 5. Boundary sidesets

Four cardinal sidesets are written to the output:

| Sideset | Exodus edge ID | Edges included |
|---|---|---|
| `south` | 1 | quad edges on the south (y-min) boundary |
| `east` | 2 | quad edges on the east (x-max) boundary |
| `north` | 3 | quad edges on the north (y-max) boundary |
| `west` | 4 | quad edges on the west (x-min) boundary |

Boundary edges are detected by edge-counting: an edge shared by only one
element is on the boundary.  Empty sidesets (e.g., an isolated single cell
where all four edges are "south", "east", "north", and "west" simultaneously)
are written correctly; sidesets with no members are omitted.

---

## Output format

The output is a standard **Exodus II** file (`NETCDF3_64BIT_OFFSET`) readable
by MOOSE, ParaView, VisIt, and the SEACAS toolchain.

| Attribute | Value |
|---|---|
| Element type | `QUAD4` |
| Spatial dimension | 3 (nodes have real x, y, z) |
| Element variables | all input variables upscaled + `THICKNESS` |
| Sidesets | `south`, `east`, `north`, `west` |
| Node ordering | CCW from above (positive Jacobian) |

---

## Running the tests

The test harness uses **pytest** and discovers test specifications from YAML
`tests` files under `test/`:

```bash
# Run all tests
python run_tests.py

# Filter by name
python run_tests.py -k simple_uniform

# Verbose output
python run_tests.py -v
```

Before running the synthetic tests for the first time, generate the fixture
`.e` files:

```bash
python test/synthetic/generate_inputs.py
```

The YAML test spec supports three test types:

| Type | What it checks |
|---|---|
| `assert` | Runs ex2ve and checks numerical properties of the output (element count, node count, variable values, sideset sizes) |
| `exodiff` | Runs ex2ve and compares the output byte-for-byte against a gold file using `exodiff` |
| `exception` | Runs ex2ve and asserts it fails with a specific error message |

See `test/synthetic/tests` for annotated examples of all three types.
