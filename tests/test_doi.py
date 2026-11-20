"""DOI columns: no placeholder depths, and a NaN-aware doi_layer.

Two things these tests hold in place (YmerFlow/ymerflow-business#124):

* A model normalized without a DOI gets NaN DOI columns. add_defaults used to
  write doi_upper = 300 and doi_lower = 500 m, and doi_layer was built from
  them, so a model that never had a DOI looked as if it had one.
* doi_layer is NaN, not 0 ("above the DOI"), for a sounding whose DOI is NaN.
"""

import numpy as np
import pandas as pd
import pytest

from libaarhusxyz.xyz import XYZ


# Layer bottoms [m below surface] shared by every sounding: 10, 30, 70, 150, inf.
DEP_BOT = [10.0, 30.0, 70.0, 150.0, np.inf]
DOI_UPPER = [25.0, 35.0, np.nan, 70.0]     # [m below surface]
DOI_LOWER = [45.0, 65.0, np.nan, np.nan]   # [m below surface]


def make_model(with_doi=False, upper=DOI_UPPER, lower=DOI_LOWER):
    """A four-sounding, five-layer resistivity model, optionally with DOI columns."""
    n = 4
    flightlines = pd.DataFrame({"line_no": [1, 1, 2, 2],
                                "x": [0.0, 10.0, 0.0, 10.0],
                                "y": [0.0, 0.0, 50.0, 50.0],
                                "topo": [100.0] * n})
    if with_doi:
        flightlines["doi_upper"] = upper
        flightlines["doi_lower"] = lower
    dep_bot = np.tile(DEP_BOT, (n, 1))
    dep_top = np.c_[np.zeros(n), dep_bot[:, :-1]]
    layer_data = {"resistivity": pd.DataFrame(np.full((n, len(DEP_BOT)), 30.0)),
                  "dep_top": pd.DataFrame(dep_top),
                  "dep_bot": pd.DataFrame(dep_bot)}
    return XYZ({"flightlines": flightlines, "layer_data": layer_data, "model_info": {}})


def doi_columns(xyz):
    return sorted(c for c in xyz.flightlines.columns if "doi" in c.lower())


# ── No placeholder DOI ───────────────────────────────────────────────────────

def test_model_without_doi_gets_nan_not_300_and_500():
    xyz = make_model()
    xyz.normalize(naming_standard="libaarhusxyz")
    for col in ("doi_upper", "doi_lower"):
        assert xyz.flightlines[col].dtype == float
        assert xyz.flightlines[col].isna().all()
    assert xyz.layer_data["doi_layer"].isna().all().all()


def test_model_without_doi_gets_no_doi_under_alc():
    xyz = make_model()
    xyz.normalize(naming_standard="alc")
    assert doi_columns(xyz) == []
    assert "doi_layer" not in xyz.layer_data


def test_existing_doi_is_kept():
    xyz = make_model(with_doi=True)
    xyz.normalize(naming_standard="libaarhusxyz")
    np.testing.assert_array_equal(xyz.flightlines["doi_upper"], DOI_UPPER)
    np.testing.assert_array_equal(xyz.flightlines["doi_lower"], DOI_LOWER)


def test_data_without_resistivity_gets_no_doi_columns():
    flightlines = pd.DataFrame({"line_no": [1, 1], "x": [0.0, 10.0], "y": [0.0, 0.0],
                                "topo": [100.0, 100.0]})
    xyz = XYZ({"flightlines": flightlines,
               "layer_data": {"dbdt_ch1gt": pd.DataFrame(np.ones((2, 3)))},
               "model_info": {}})
    xyz.normalize(naming_standard="libaarhusxyz")
    assert doi_columns(xyz) == []


# ── doi_layer ────────────────────────────────────────────────────────────────

# Layer bottoms 10, 30, 70, 150, inf against each sounding's DOI_UPPER / DOI_LOWER.
EXPECTED_DOI_LAYER = np.array([
    [0, 1, 2, 2, 2],                       # upper 25, lower 45
    [0, 0, 2, 2, 2],                       # upper 35, lower 65
    [np.nan] * 5,                          # both NaN
    [np.nan] * 5,                          # upper 70, lower NaN
])


@pytest.mark.parametrize("naming_standard", ["libaarhusxyz", "alc"])
def test_doi_layer_is_nan_where_the_doi_is_nan(naming_standard):
    xyz = make_model(with_doi=True)
    xyz.normalize(naming_standard=naming_standard)
    np.testing.assert_array_equal(xyz.layer_data["doi_layer"].to_numpy(), EXPECTED_DOI_LAYER)


def test_doi_layer_boundaries():
    """A layer whose bottom sits exactly on a DOI counts as above it."""
    xyz = make_model(with_doi=True, upper=[10.0] * 4, lower=[70.0] * 4)
    xyz.normalize(naming_standard="libaarhusxyz")
    np.testing.assert_array_equal(xyz.layer_data["doi_layer"].iloc[0].to_numpy(), [0, 1, 1, 2, 2])
