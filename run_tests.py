#!/usr/bin/env python
"""ex2ve test harness.

Discovers ``tests`` YAML files under ``test/**`` and runs each named entry as a
pytest case. Three test types are supported:

    type: assert     - run ex2ve, then check numerical properties of the output
                       Exodus file against analytical expected values.
                       Supports these assertion keys:
                           num_elem            (int)
                           num_nodes           (int)
                           elem_vars:          (dict of var_name -> scalar or list)
                               Scalar means every output element should equal this.
                               List means per-element values in output order.
                           elem_vars_sorted:   (dict of var_name -> list)
                               Checks the SORTED set of values, useful when the
                               output element order is not pinned by the test.
                           sidesets:           (dict of sideset_name -> int)
                               Expected count of sides in each named sideset.
                               Sideset names: south, east, north, west.
                       Tolerances: rtol (default 1e-6), atol (default 1e-6).

    type: exodiff    - run ex2ve, then exodiff the output against a gold file.
                       Requires: gold (filename, relative to <test_dir>/gold/).

    type: exception  - run ex2ve and assert it errors with a specific message.
                       Requires: expected_error (substring of the error message).

Common keys for all types:
    filename         input file (relative to the directory containing 'tests')
    cli_args         extra CLI flags to pass to ex2ve

Usage:
    python run_tests.py                    # run all tests
    python run_tests.py -k simple_uniform  # filter by name
    python run_tests.py --exodiff /path/to/pyexodiff.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from netCDF4 import Dataset, chartostring


ROOT = Path(__file__).resolve().parent
EX2VE = ROOT / "ex2ve.py"


def _discover_tests() -> dict[str, dict]:
    tests: dict[str, dict] = {}
    for root, _dirs, files in os.walk(ROOT / "test", topdown=False):
        if "tests" not in files:
            continue
        test_file = Path(root) / "tests"
        with open(test_file) as f:
            cfg = yaml.safe_load(f) or {}
        for key, values in cfg.items():
            full_key = f"{Path(root).relative_to(ROOT)}/{key}"
            entry = dict(values)
            entry["_dir"] = Path(root)
            tests[full_key] = entry
    return tests


TESTS = _discover_tests()


def _decode_names(name_var) -> list[str]:
    return [str(s).strip() for s in chartostring(name_var[:])]


def _read_output(
    path: Path,
) -> tuple[int, int, dict[str, np.ndarray], dict[str, int]]:
    """Return (num_elem, num_nodes, {var_name: values}, {sideset_name: count})."""
    with Dataset(str(path), "r") as ds:
        num_elem = ds.dimensions["num_elem"].size
        num_nodes = ds.dimensions["num_nodes"].size
        elem_vars: dict[str, np.ndarray] = {}
        if (
            "name_elem_var" in ds.variables
            and "time_step" in ds.dimensions
            and ds.dimensions["time_step"].size > 0
        ):
            names = _decode_names(ds.variables["name_elem_var"])
            for v_idx, name in enumerate(names, start=1):
                var_name = f"vals_elem_var{v_idx}eb1"
                if var_name in ds.variables:
                    elem_vars[name] = np.asarray(ds.variables[var_name][-1, :])

        sidesets: dict[str, int] = {}
        if "num_side_sets" in ds.dimensions:
            n_ss = ds.dimensions["num_side_sets"].size
            names = (
                _decode_names(ds.variables["ss_names"])
                if "ss_names" in ds.variables else [f"ss{i + 1}" for i in range(n_ss)]
            )
            for i in range(1, n_ss + 1):
                dim_name = f"num_side_ss{i}"
                if dim_name in ds.dimensions:
                    sidesets[names[i - 1]] = ds.dimensions[dim_name].size
    return num_elem, num_nodes, elem_vars, sidesets


def _build_command(spec: dict, input_path: Path) -> list[str]:
    cmd = [sys.executable, str(EX2VE), str(input_path)]
    if "cli_args" in spec:
        cmd.extend(str(spec["cli_args"]).split())
    return cmd


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_ve.e")


@pytest.mark.parametrize("key", TESTS)
def test_ex2ve(key, exodiff):
    """Drive each YAML-declared test case."""
    spec = TESTS[key]

    if "type" not in spec:
        pytest.skip(f"{key}: 'type' not specified")
    if "filename" not in spec:
        pytest.skip(f"{key}: 'filename' not specified")

    test_dir: Path = spec["_dir"]
    input_path = test_dir / spec["filename"]
    output_path = _default_output_path(input_path)

    test_type = spec["type"]
    if test_type == "assert":
        _run_assert(key, spec, input_path, output_path)
    elif test_type == "exodiff":
        if "gold" not in spec:
            pytest.skip(f"{key}: 'gold' not specified")
        _run_exodiff(key, spec, input_path, output_path, exodiff)
    elif test_type == "exception":
        if "expected_error" not in spec:
            pytest.skip(f"{key}: 'expected_error' not specified")
        _run_exception(key, spec, input_path)
    else:
        pytest.skip(f"{key}: unknown test type {test_type!r}")


def _run_assert(key: str, spec: dict, input_path: Path, output_path: Path) -> None:
    cmd = _build_command(spec, input_path)
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{key}: ex2ve exited non-zero\n{exc.output}")

    num_elem, num_nodes, elem_vars, sidesets = _read_output(output_path)

    rtol = float(spec.get("rtol", 1e-6))
    atol = float(spec.get("atol", 1e-6))
    assertions: dict = spec.get("assertions", {})

    if "num_elem" in assertions:
        expected = int(assertions["num_elem"])
        assert num_elem == expected, (
            f"{key}: num_elem={num_elem}, expected {expected}"
        )

    if "num_nodes" in assertions:
        expected = int(assertions["num_nodes"])
        assert num_nodes == expected, (
            f"{key}: num_nodes={num_nodes}, expected {expected}"
        )

    for var, expected in assertions.get("elem_vars", {}).items():
        assert var in elem_vars, f"{key}: variable {var!r} missing from output"
        actual = elem_vars[var]
        expected_arr = np.asarray(expected, dtype=float)
        if expected_arr.shape == ():
            expected_arr = np.full(num_elem, float(expected_arr))
        assert actual.shape == expected_arr.shape, (
            f"{key}/{var}: shape mismatch {actual.shape} vs {expected_arr.shape}"
        )
        np.testing.assert_allclose(
            actual, expected_arr, rtol=rtol, atol=atol,
            err_msg=f"{key}/{var}: values disagree",
        )

    for var, expected in assertions.get("elem_vars_sorted", {}).items():
        assert var in elem_vars, f"{key}: variable {var!r} missing from output"
        actual_sorted = np.sort(elem_vars[var])
        expected_sorted = np.sort(np.asarray(expected, dtype=float))
        np.testing.assert_allclose(
            actual_sorted, expected_sorted, rtol=rtol, atol=atol,
            err_msg=f"{key}/{var}: sorted values disagree",
        )

    for name, expected_count in assertions.get("sidesets", {}).items():
        assert name in sidesets, (
            f"{key}: sideset {name!r} missing from output "
            f"(present: {sorted(sidesets.keys())})"
        )
        assert sidesets[name] == int(expected_count), (
            f"{key}: sideset {name!r} has {sidesets[name]} sides, "
            f"expected {expected_count}"
        )


def _run_exodiff(
    key: str, spec: dict, input_path: Path, output_path: Path, exodiff_cmd: str,
) -> None:
    cmd = _build_command(spec, input_path)
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{key}: ex2ve exited non-zero\n{exc.output}")

    gold_path = spec["_dir"] / "gold" / spec["gold"]
    try:
        subprocess.check_output(
            [exodiff_cmd, "--quiet", str(output_path), str(gold_path)],
            stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError:
        pytest.skip(f"{key}: exodiff binary {exodiff_cmd!r} not found")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{key}: exodiff reported differences\n{exc.output}")


def _run_exception(key: str, spec: dict, input_path: Path) -> None:
    cmd = _build_command(spec, input_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = (result.stdout or "") + (result.stderr or "")
    expected = str(spec["expected_error"])
    assert expected in combined, (
        f"{key}: expected error substring {expected!r} not found in output:\n{combined}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-rsx", "--tb=short", __file__, *sys.argv[1:]]))
