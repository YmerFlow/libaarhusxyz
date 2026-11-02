# Changelog

## 2026-10-31

### VTK and GLB export from inversion outputs (YmerFlow/ymerflow-business#135)

Both exporters now prepare a copy of the model with `export.vtk._export_model` before writing; the model passed in is left unchanged (they used to overwrite its `dep_bot` and add columns to its flightlines).

- The copy uses libaarhusxyz names whatever the model's naming standard, so an alc-named inversion output (`rho`, `UTMX`, `Topography`, `Line`, ...) exports its resistivity, coordinates and line. `attr_out` names in any standard are mapped the same way. Default attributes the model does not have are skipped silently instead of one printed line each; the default list has `tx_altitude` where it had `tx_alt` and no longer lists `line_id` (it maps to `title`).
- Soundings without a finite layer geometry (never inverted: NaN resistivity, infinite depths) are left out of the mesh. They used to put `inf` into the VTK points and break the GLB.
- An infinite last-layer bottom (the halfspace) becomes its top plus the thickness of the layer above, per sounding. The VTK title line and the GLB `asset.extras` say so. The old code used one value from row 0 for every sounding and failed with `KeyError: 0` on data whose index did not start at 0.
- Datasets without layer depths (AEM data) raise `ValueError("No layer depths ...")` instead of `KeyError: 'dep_bot'`.
- GLB: write `b"".join(gltf.save_to_bytes())`; pygltflib returns the file as a list of chunks, so the writer could not have written a file before.

## 2026-01-29

### Added case-insensitive column name detection in `xyz.py`

Added a `_case_variants()` helper function that generates lowercase, uppercase, and title-case variants of column names. Updated all column-detection properties to use this helper:

- `line_id_column`
- `x_column`
- `y_column`
- `z_column`
- `tilt_roll_column`
- `tilt_pitch_column`
- `alt_column`

This allows files with uppercase column names like `X`, `Y`, `Z` to be detected automatically without requiring an ALC file.

### Added `extra_mappings` parameter for custom column name normalization

Added support for user-provided column name mappings during normalization, similar to how `alcfile` works for parsing.

**Files modified:**
- `normalizer.py`: Updated `get_name_mapper()`, `normalize_headers()`, `normalize_column_names()`, `normalize_naming()`, and `normalize()` to accept `extra_mappings` parameter
- `xyz.py`: Updated class docstring and `normalize_naming()` method

**Usage:**
```python
import libaarhusxyz

# Dictionary form
xyz = libaarhusxyz.XYZ(
    "model.xyz",
    normalize=True,
    extra_mappings={"Res": "resistivity", "Thick": "height"}
)

# CSV file form
xyz = libaarhusxyz.XYZ(
    "model.xyz",
    normalize=True,
    extra_mappings="/path/to/custom_mappings.csv"
)

# Or call normalize separately
xyz = libaarhusxyz.XYZ("model.xyz")
xyz.normalize(extra_mappings={"Res": "resistivity", "Thick": "height"})
```

The `extra_mappings` parameter accepts:
- `dict`: `{input_name: canonical_name}` mappings
- `str`: path to a CSV file with columns `libaarhusxyz` and `input`
- `DataFrame`: with columns `libaarhusxyz` and `input`

Custom mappings take precedence over default mappings from `normalizer.csv`.

### change default calculations of some layer_data

Changes to libaarhusxyz.normalizer.add_defaults and libaarhusxyz.normalizer.normalize_depths affect their calculaiton, 
including:
- `height` set to `numpy.nan` when not present
- `dep_bot` set to the cummulative sum of `height`s when the latter is present
 