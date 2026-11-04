"""normalize() must keep an xdist the flightlines already carry.

xdist is the distance along the flight path. It is measured on the full-rate
data and carried through averaging and decimation. Recomputing it afterwards
from the remaining soundings sums straight chords between them, which cuts
the corners of the real path and shortens every line. An inversion normalizes
its input and each output dataset, so its model and synthetic came out
shorter than the data they were inverted from, and features drifted along the
line when the two were plotted against xdist.
"""

import numpy as np
import pandas as pd
import pytest

from libaarhusxyz.xyz import XYZ


def decimated_line(title, y0=0.0, every=10):
    """Flightlines for one line, decimated from a full-rate path that weaves sideways.

    The full-rate soundings are 10 m apart along x and weave +-20 m in y.
    xdist [m] is measured along that path, then every `every`-th sounding is
    kept, as averaging or decimation would.
    """
    x = np.arange(1000) * 10.0
    y = y0 + 20.0 * np.sin(2 * np.pi * x / 500.0)
    xdist = np.cumsum(np.append([0.0], np.hypot(np.diff(x), np.diff(y))))
    keep = slice(None, None, every)
    n = len(x[keep])
    return pd.DataFrame({
        "title": np.full(n, float(title)),
        "x": x[keep],
        "y": y[keep],
        "xdist": xdist[keep],
        "topo": np.zeros(n),
        "tx_altitude": np.full(n, 30.0),
    })


def two_lines():
    """Flightlines for two decimated lines, xdist [m] restarting at 0 on each."""
    return pd.concat([decimated_line(1), decimated_line(2, y0=500.0)],
                     ignore_index=True)


def chord_xdist(fl, x="x", y="y", title="title"):
    """Cumulative straight-line distance [m] between consecutive soundings, per line."""
    out = []
    for _, line in fl.groupby(title, sort=False):
        step = np.hypot(np.diff(line[x].values), np.diff(line[y].values))
        out.append(np.cumsum(np.append([0.0], step)))
    return np.concatenate(out)


def data_xyz(flightlines):
    """An XYZ holding three gates of dB/dt data for `flightlines`."""
    n = len(flightlines)
    return XYZ({"flightlines": flightlines.copy(),
                "model_info": {},
                "layer_data": {"dbdt_ch1gt": pd.DataFrame(np.ones((n, 3)))}})


def model_xyz(flightlines):
    """An XYZ holding a three-layer model, 10 ohm.m, for `flightlines`."""
    n = len(flightlines)
    return XYZ({"flightlines": flightlines.copy(),
                "model_info": {},
                "layer_data": {
                    "resistivity": pd.DataFrame(np.full((n, 3), 10.0)),
                    "dep_top": pd.DataFrame(np.tile([0.0, 10.0, 20.0], (n, 1))),
                    "dep_bot": pd.DataFrame(np.tile([10.0, 20.0, np.nan], (n, 1))),
                }})


def test_the_given_xdist_is_longer_than_recomputing_it():
    # Guards the tests below: if the fixture's xdist equalled the chord
    # distance, keeping it and recomputing it would look the same.
    fl = two_lines()
    past_start = fl["xdist"].values > 0
    assert np.all(fl["xdist"].values[past_start] > chord_xdist(fl)[past_start])


def test_data_keeps_its_xdist():
    fl = two_lines()
    data = data_xyz(fl)
    data.normalize(naming_standard="libaarhusxyz")
    np.testing.assert_array_equal(data.flightlines["xdist"].values, fl["xdist"].values)


def test_model_and_synthetic_keep_the_input_xdist():
    # The inversion's path: normalize the input, build the model and the
    # synthetic on a copy of its flightlines (as simpeg's static_instrument
    # inverted_model_to_xyz and forward_data_to_xyz do), normalize each to
    # the alc naming standard and write it out.
    fl = two_lines()
    data = data_xyz(fl)
    data.normalize(naming_standard="libaarhusxyz")

    model = model_xyz(data.flightlines)
    model.normalize(naming_standard="alc")
    synthetic = data_xyz(data.flightlines)
    synthetic.normalize(naming_standard="alc")

    np.testing.assert_array_equal(model.flightlines["xdist"].values, fl["xdist"].values)
    np.testing.assert_array_equal(synthetic.flightlines["xdist"].values, fl["xdist"].values)


@pytest.mark.parametrize("missing", ["absent", "all NaN"])
def test_missing_xdist_is_computed(missing):
    fl = two_lines()
    if missing == "absent":
        fl = fl.drop(columns="xdist")
    else:
        fl["xdist"] = np.nan
    data = data_xyz(fl)
    data.normalize(naming_standard="libaarhusxyz")
    np.testing.assert_allclose(data.flightlines["xdist"].values, chord_xdist(fl))


def test_reprojection_recomputes_xdist():
    # Kept across a change of CRS, xdist would stay in the old CRS's units.
    fl = two_lines()
    fl["x"] += 400000.0
    fl["y"] += 3800000.0
    data = data_xyz(fl)
    data.model_info["projection"] = 32613
    data.normalize(naming_standard="libaarhusxyz", project_crs=32612)

    out = data.flightlines
    np.testing.assert_allclose(out["xdist"].values, chord_xdist(out))
    assert not np.allclose(out["xdist"].values, fl["xdist"].values)
