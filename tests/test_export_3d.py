"""VTK and GLB export from an inversion output (YmerFlow/ymerflow-business#135).

An inversion output, as the platform saves it, is in alc names (rho, UTMX,
Topography, Line, ...), ends in a halfspace whose dep_bot is infinite, and
carries soundings that were never inverted: NaN resistivity and infinite
depths in every layer. Both exporters used to fail on it. The VTK writer put
inf into the point coordinates, overwrote the dataset's dep_bot in place, and
printed a line for every default attribute it could not find under its
libaarhusxyz name. The GLB writer could not put the infinities into its JSON
header. AEM data (gates, no layer depths) failed with KeyError 'dep_bot'.
"""

import io
import os.path

import numpy as np
import pandas as pd
import pytest
from pygltflib import GLTF2

from libaarhusxyz.export import vtk
from libaarhusxyz.xyz import XYZ


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WORKBENCH_MODEL = os.path.join(
    DATA_DIR, "aarhus_workbench.6.7.0.0", "SCI_1_Pro3_MOD_inv_example_SCI_inversion_export.xyz")

# Layer tops [m below surface] per sounding; thicknesses differ between soundings.
DEP_TOP = np.array([[0.0, 10.0, 30.0, 60.0],
                    [0.0, 10.0, 30.0, 60.0],
                    [0.0, 5.0, 15.0, 35.0],
                    [0.0, 5.0, 15.0, 35.0],
                    [np.inf] * 4,            # never inverted
                    [0.0, 8.0, 20.0, 44.0]])
UNMODELED = 4


def inversion_output():
    """Six soundings on two lines in alc names, as invert_tem saves a model."""
    n, nl = DEP_TOP.shape
    flightlines = pd.DataFrame({"Line": [100, 100, 100, 200, 200, 200],
                                "UTMX": [0.0, 20.0, 40.0, 0.0, 20.0, 40.0],
                                "UTMY": [0.0, 0.0, 0.0, 500.0, 500.0, 500.0],
                                "Topography": [100.0, 101.0, 102.0, 90.0, 91.0, 92.0],
                                "TxAltitude": [30.0] * n,
                                "resdata": [1.0] * n})
    rho = np.full((n, nl), 50.0)
    rho[UNMODELED] = np.nan
    dep_bot = np.c_[DEP_TOP[:, 1:], np.full(n, np.inf)]
    xyz = XYZ({"flightlines": flightlines,
               "layer_data": {"rho": pd.DataFrame(rho),
                              "dep_top": pd.DataFrame(DEP_TOP.copy()),
                              "dep_bot": pd.DataFrame(dep_bot)},
               "model_info": {}})
    xyz.normalize(naming_standard="alc")
    return xyz


def write_vtk(xyz):
    buf = io.StringIO()
    xyz.to_vtk(buf)
    return buf.getvalue()


def write_glb(xyz):
    buf = io.BytesIO()
    xyz.to_glb(buf)
    return GLTF2.load_from_bytes(buf.getvalue())


def vtk_points(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    count = int(lines[start].split()[1])
    return np.array([line.split() for line in lines[start + 1:start + 1 + count]], dtype=float)


def vtk_scalars(text):
    return [line.split()[1] for line in text.splitlines() if line.startswith("SCALARS")]


# ── The inversion output exports ─────────────────────────────────────────────

def test_vtk_from_an_inversion_output_has_finite_points_and_the_attributes():
    text = write_vtk(inversion_output())
    assert np.isfinite(vtk_points(text)).all()
    assert vtk.HALFSPACE_NOTE in text.splitlines()[1]
    assert {"resistivity", "x", "y", "topo", "title", "dep_top", "dep_bot",
            "tx_altitude", "resdata", "xdist"} <= set(vtk_scalars(text))


def test_glb_from_an_inversion_output_is_a_valid_file():
    gltf = write_glb(inversion_output())
    position = gltf.accessors[0]
    assert position.count > 0
    assert np.isfinite(position.min + position.max).all()
    assert gltf.asset.extras == {"note": vtk.HALFSPACE_NOTE}


def test_export_prints_nothing_for_absent_attributes(capsys):
    xyz = inversion_output()
    capsys.readouterr()
    write_vtk(xyz)
    write_glb(xyz)
    assert capsys.readouterr().out == ""


def test_export_leaves_the_dataset_unchanged():
    xyz = inversion_output()
    columns = list(xyz.flightlines.columns)
    write_vtk(xyz)
    write_glb(xyz)
    assert list(xyz.flightlines.columns) == columns
    assert np.isinf(xyz.layer_data["dep_bot"].iloc[:, -1]).all()
    assert np.isinf(xyz.layer_data["dep_top"].iloc[UNMODELED]).all()


def test_each_line_part_exports():
    """write_dataset also exports every line on its own."""
    for part in inversion_output().split_by_line().values():
        assert np.isfinite(vtk_points(write_vtk(part))).all()
        write_glb(part)


# ── What the export copy holds ───────────────────────────────────────────────

def test_halfspace_bottom_is_its_top_plus_the_layer_above():
    export, _, note = vtk._export_model(inversion_output(), vtk.DEFAULT_ATTR_OUT)
    tops = np.delete(DEP_TOP, UNMODELED, axis=0)
    expected = tops[:, -1] + (tops[:, -1] - tops[:, -2])        # 90, 90, 55, 55, 68 m
    np.testing.assert_array_equal(export.layer_data["dep_bot"].iloc[:, -1], expected)
    assert note == vtk.HALFSPACE_NOTE


def test_soundings_never_inverted_are_left_out():
    xyz = inversion_output()
    export, _, _ = vtk._export_model(xyz, vtk.DEFAULT_ATTR_OUT)
    assert len(export.flightlines) == len(xyz.flightlines) - 1
    assert all(len(ldf) == len(export.flightlines) for ldf in export.layer_data.values())
    text = write_vtk(xyz)
    n_cells = int(next(line for line in text.splitlines() if line.startswith("CELLS")).split()[1])
    assert n_cells == (len(xyz.flightlines) - 1) * DEP_TOP.shape[1]


def test_attr_out_names_are_mapped_to_libaarhusxyz():
    _, attr_out, _ = vtk._export_model(inversion_output(), ["rho", "UTMX", "line_id", "title"])
    assert attr_out == ["resistivity", "x", "title"]


def test_finite_model_gets_no_note():
    xyz = inversion_output()
    xyz.layer_data["dep_bot"].iloc[:, -1] = 200.0
    text = write_vtk(xyz)
    assert vtk.HALFSPACE_NOTE not in text
    assert write_glb(xyz).asset.extras in ({}, None)


# ── Other inputs ─────────────────────────────────────────────────────────────

def test_aem_data_raises_a_clear_error():
    flightlines = pd.DataFrame({"Line": [100, 100], "UTMX": [0.0, 20.0], "UTMY": [0.0, 0.0],
                                "Topography": [100.0, 100.0]})
    data = XYZ({"flightlines": flightlines,
                "layer_data": {"Gate_Ch01": pd.DataFrame(np.ones((2, 3)))},
                "model_info": {}})
    with pytest.raises(ValueError, match="No layer depths"):
        write_vtk(data)
    with pytest.raises(ValueError, match="No layer depths"):
        write_glb(data)


def test_workbench_model_exports():
    xyz = XYZ(WORKBENCH_MODEL, normalize=True)
    assert np.isfinite(vtk_points(write_vtk(xyz))).all()
    write_glb(xyz)
