"""Comments in GEX files, and the line before a section header.

Two conventions exist. NRG's Xcite description comments with ``#`` after values
and on headers; Aarhus/SkyTEM files comment whole lines with ``/``. Neither may
reach the number parser, and a section's last line must survive with no blank
line before the next header.
"""
import numpy as np
import libaarhusxyz


COMMENTED = """\
/ preamble line, SkyTEM style, kept as the header
[General]
## full-line hash comment
Description=Test system # inline comment on a string
TxLoopArea=265.9 # transmitter area in m^2
NumberOfTurns=4
/ full-line slash comment inside a section
TxLoopPoint1= 9.20 0.00 # only the first point is commented
TxLoopPoint2= 0.00 9.20
TxLoopPoint3= -9.20 0.00
WaveformPoint01= 0.0 0.0 # commented
WaveformPoint02= 1.0e-3 1.0
GateTime01= 1e-5 0.5e-5 1.5e-5
GateTime02= 2e-5 1.5e-5 2.5e-5
[Channel1] # channel 1 is the z receiver
NoGates=2
TxApproximateCurrent=280
ReceiverPolarizationXYZ=Z
[Channel2] # no blank line before this header either
NoGates=2
TxApproximateCurrent=280
ReceiverPolarizationXYZ=X
"""


def _parse(tmp_path):
    path = tmp_path / "commented.gex"
    path.write_text(COMMENTED)
    return libaarhusxyz.GEX(str(path)).gex_dict


def test_inline_hash_comments_do_not_break_arrays(tmp_path):
    gex = _parse(tmp_path)
    general = gex["General"]
    assert np.asarray(general["TxLoopPoint"]).shape == (3, 2)
    assert np.asarray(general["WaveformPoint"]).shape == (2, 2)
    assert np.asarray(general["GateTime"]).shape == (2, 3)
    assert general["TxLoopArea"] == 265.9


def test_header_comments_leave_clean_section_names(tmp_path):
    gex = _parse(tmp_path)
    assert "Channel1" in gex and "Channel2" in gex
    assert gex["Channel1"]["ReceiverPolarizationXYZ"] == "Z"
    assert gex["Channel2"]["ReceiverPolarizationXYZ"] == "X"


def test_slash_comment_inside_section_is_ignored_and_preamble_kept(tmp_path):
    gex = _parse(tmp_path)
    assert "/ full-line slash comment" not in str(gex["General"])
    assert gex["header"].startswith("/ preamble")


def test_last_line_before_a_header_is_kept(tmp_path):
    gex = _parse(tmp_path)
    # GateTime02 is the last line before [Channel1]; ReceiverPolarizationXYZ the
    # last before [Channel2]. Neither has a blank line after it.
    assert np.asarray(gex["General"]["GateTime"]).shape[0] == 2
    assert gex["Channel1"]["ReceiverPolarizationXYZ"] == "Z"
    assert gex["Channel2"]["NoGates"] == 2
