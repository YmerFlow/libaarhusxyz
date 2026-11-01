import pandas as pd
import libaarhusxyz
import numpy as np

from scipy.interpolate import interp1d
from datetime import datetime

from .. import normalizer

HALFSPACE_NOTE = ("Last-layer bottom (halfspace) set for display only: "
                  "its top plus the thickness of the layer above.")

DEFAULT_ATTR_OUT = ['resistivity', 'resistivity_variance_factor', 'title', 'x', 'y',
                    'topo', 'dep_top', 'dep_bot', 'tx_altitude', 'invalt', 'invaltstd',
                    'deltaalt', 'numdata', 'resdata',
                    'restotal', 'doi_upper', 'doi_lower', 'xdist']

def _export_model(model, attr_out):
    """Return a copy of `model` ready for 3D export, its attribute names, and a metadata note.

    The copy uses libaarhusxyz names (resistivity, x, y, topo, title, ...)
    whatever naming standard `model` is in, drops soundings without a finite
    layer geometry (soundings that were never inverted), and gives a
    non-finite last-layer bottom (the halfspace) a finite depth: its top plus
    the thickness of the layer above. `model` itself is left unchanged.

    Parameters
    ----------
    model : XYZ
        Resistivity model with layer depths: dep_top / dep_bot [m below the
        surface] or height [m].
    attr_out : list of str
        Attributes to write, in any naming standard.

    Returns
    -------
    export : XYZ
        The copy described above.
    attr_out : list of str
        `attr_out` in libaarhusxyz names, without duplicates.
    note : str or None
        HALFSPACE_NOTE when a last-layer bottom was replaced, else None.

    Raises
    ------
    ValueError
        If `model` has no layer depths (AEM data rather than a model), or no
        sounding has a finite layer geometry.
    """
    export = libaarhusxyz.XYZ({
        "flightlines": model.flightlines.copy(),
        "layer_data": {key: ldf.copy() for key, ldf in model.layer_data.items()},
        "model_info": dict(model.model_info)})
    normalizer.normalize_naming(export, "libaarhusxyz")
    export.flightlines = export.flightlines.loc[:, ~export.flightlines.columns.duplicated()]
    normalizer.normalize_depths(export)
    layers = export.layer_data
    if "dep_top" not in layers or "dep_bot" not in layers:
        raise ValueError("No layer depths (dep_top / dep_bot): only resistivity models can be "
                         "exported in 3D, not AEM data.")

    dep_top = layers["dep_top"].to_numpy(dtype=float)
    dep_bot = layers["dep_bot"].to_numpy(dtype=float)
    keep = np.isfinite(dep_top).all(axis=1) & np.isfinite(dep_bot[:, :-1]).all(axis=1)
    if not keep.any():
        raise ValueError("No sounding has a finite layer geometry to export.")
    export.flightlines = export.flightlines.loc[keep].reset_index(drop=True)
    for key in list(layers):
        layers[key] = layers[key].loc[keep].reset_index(drop=True)

    note = None
    dep_top, dep_bot = dep_top[keep], dep_bot[keep]
    halfspace = ~np.isfinite(dep_bot[:, -1])
    if halfspace.any():
        if dep_top.shape[1] < 2:
            raise ValueError("A single halfspace layer has no finite bottom to export.")
        thickness_above = dep_top[:, -1] - dep_top[:, -2]
        layers["dep_bot"].iloc[:, -1] = np.where(halfspace, dep_top[:, -1] + thickness_above,
                                                 dep_bot[:, -1])
        note = HALFSPACE_NOTE

    mapper = normalizer.get_name_mapper("libaarhusxyz")
    attr_out = list(dict.fromkeys(mapper(attr) for attr in attr_out))
    return export, attr_out, note

def _compute_xdist(fl, x_col, y_col, line_id_col):
    fl["prevdist"] = np.append(
        [0],
        np.sqrt(  (fl[x_col].values[1:] - fl[x_col].values[:-1])**2
                + (fl[y_col].values[1:] - fl[y_col].values[:-1])**2))
    fl.loc[np.append([False], fl[line_id_col].values[1:] != fl[line_id_col].values[:-1]), "prevdist"] = 0
    fl["xdist"] = fl.groupby(fl[line_id_col])["prevdist"].cumsum()

def _compute_sounding_widths(fl, line_id_col):
    #compute appropriate start and end points along line in terms of xdistance
    fl["next_title"] = fl[line_id_col].shift(-1).fillna(-1)
    fl["prev_title"] = fl[line_id_col].shift(1).fillna(-1)
    fl["next_xdist"] = fl["xdist"].shift(-1).fillna(fl.iloc[-1]["xdist"])
    fl["prev_xdist"] = fl["xdist"].shift(1).fillna(0)

    fl.loc[fl[line_id_col] != fl["prev_title"], "prev_xdist"] = fl.loc[fl[line_id_col] != fl["prev_title"], "xdist"]
    fl.loc[fl[line_id_col] != fl["next_title"], "next_xdist"] = fl.loc[fl[line_id_col] != fl["next_title"], "xdist"]

    fl.loc[fl["xdist"] - fl["prev_xdist"] > 50, "prev_xdist"] = fl.loc[fl["xdist"] - fl["prev_xdist"] > 50, "xdist"] - 50
    fl.loc[fl["next_xdist"] - fl["xdist"] > 50, "next_xdist"] = fl.loc[fl["next_xdist"] - fl["xdist"] > 50, "xdist"] + 50

    fl["left"] = (fl["prev_xdist"] + fl["xdist"]) / 2
    fl["right"] = (fl["xdist"] + fl["next_xdist"]) / 2

def _generate_cells(fl, df, x_col, y_col, z_col, line_id_col):
    #convert xdist along line to appropriate X, Y, and topo values
    titles = fl[line_id_col].unique()
    columns = [(x_col, 'x'), (y_col, 'y'), (z_col, 'topo')]

    for ln in titles:
        mask_ln = fl[line_id_col] == ln
        xdist = fl.loc[mask_ln, 'xdist'].values

        for src_col, dest_prefix in columns:
            col_vals = fl.loc[mask_ln, src_col].values
            interp_func = interp1d(xdist, col_vals, fill_value=(col_vals[0],col_vals[-1]))
            fl.loc[mask_ln, dest_prefix+'_left'] = interp_func(fl.loc[mask_ln, 'left'])
            fl.loc[mask_ln, dest_prefix+'_right'] = interp_func(fl.loc[mask_ln, 'right'])

    cells = df.merge(fl, left_on='record', right_index=True)
    cells.loc[:,'z_bot_left'] = cells.topo_left-cells.dep_bot
    cells.loc[:,'z_bot_right'] = cells.topo_right-cells.dep_bot
    cells.loc[:,'z_top_left'] = cells.topo_left-cells.dep_top
    cells.loc[:,'z_top_right'] = cells.topo_right-cells.dep_top

    return cells
    
def _generate_points_array(cells):
    corner_vertices = pd.DataFrame(
        data={
        'x':['x_left','x_left','x_right','x_right'],
        'y':['y_left','y_left','y_right','y_right'],
        'z':['z_top_left','z_bot_left','z_bot_right','z_top_right']
    }, index = [0,1,2,3])

    cell_vertices = np.zeros((cells.shape[0],3,4))
    for n in range(4):
        cell_vertices[:,0,n] = cells.loc[:,  corner_vertices.loc[n,'x'] ] 
        cell_vertices[:,1,n] = cells.loc[:,  corner_vertices.loc[n,'y'] ] 
        cell_vertices[:,2,n] = cells.loc[:,  corner_vertices.loc[n,'z'] ]
    
    return np.swapaxes(cell_vertices,1,2).reshape((cells.shape[0]*4,3))

def _write_vtk(point_coordinates,
              cell_indices_np, cells_out_vtk, cell_types_out_vtk,
              cells,
              fid,
              attr_out,
              note=None):
    print('# vtk DataFile Version 2.0', file=fid)
    # print('AEM grid export of tin_id = %s on %s ' %(tri['meta']['tri_id'], datetime.now().isoformat()), file=fid)
    title = 'AEM grid export of TIN on %s' % (datetime.now().isoformat(),)
    if note:
        title += '. ' + note
    print(title, file=fid)
    print('ASCII', file=fid)

    fid.write('DATASET UNSTRUCTURED_GRID\n')

    print('POINTS', point_coordinates.shape[0], 'float', file=fid)
    np.savetxt(fid, point_coordinates, fmt='%.2f', delimiter=' ', newline='\n', )

    print('CELLS', cell_indices_np.shape[0], cells_out_vtk.size, file=fid)
    np.savetxt(fid, cells_out_vtk, fmt='%d', delimiter=' ', newline='\n')

    print('CELL_TYPES', cell_indices_np.shape[0], file=fid)
    np.savetxt(fid, cell_types_out_vtk, fmt='%d', delimiter=' ', newline='\n')

    print('CELL_DATA', cells.shape[0], file=fid)
    for attr in attr_out:
        if attr not in cells.columns.to_list():
            # Absent from this model (e.g. Workbench-only columns): nothing to write.
            continue
        elif not np.issubdtype(cells.dtypes[attr],np.number):
            print(attr, ' is not a numeric column. this attribute will not be written to the VTK.')
            continue
        else:
            datatype = 'float'

            # if np.issubdtype(p.dtypes[attr],np.integer):
            #     datatype = 'long'
            attr_alphanumeric = "".join([ c if c.isalnum() else "_" for c in attr ])

            print('SCALARS', attr_alphanumeric, datatype,'1', file=fid)
            print('LOOKUP_TABLE default', file=fid)
            np.savetxt(fid, cells.loc[:, attr].to_numpy(), fmt='%f', delimiter=' ', newline='\n')

def _flatten_layer_data(model):
    df_list = []
    for key, val in model.layer_data.items():
        df_temp = model.layer_data[key].stack()
        df_temp.index.rename(('record','layer'), inplace=True)
        df_temp.rename(key, inplace=True)
        df_list.append(df_temp)
    return pd.concat(df_list, axis=1)

def _vtk_cell_data(points_array):
    indices_points = np.arange(points_array.size).reshape([-1,4])
    unique, unique_indices, unique_inverse = np.unique(np.around(points_array,3), True, True, axis=0)
    indices_points_unique = unique_inverse.reshape([-1,4])

    point_coordinates = unique

    cell_indices_np = indices_points_unique
    num_nodes = np.ones( (cell_indices_np.shape[0],1), dtype=int)*4

    cells_out_vtk = np.concatenate( (num_nodes,cell_indices_np),axis=1)
    cell_types_out_vtk = np.ones( (cell_indices_np.shape[0],1), dtype=int)*9 # 9 is for VTK_QUAD cell type

    return point_coordinates, cell_indices_np, cells_out_vtk, cell_types_out_vtk

def _dump(model, fid, attr_out=DEFAULT_ATTR_OUT):
    model, attr_out, note = _export_model(model, attr_out)
    fl = model.flightlines

    # Get column names based on current naming standard
    x_col = model.x_column
    y_col = model.y_column
    z_col = model.z_column

    if not x_col or not y_col or not z_col:
        raise ValueError(f"Missing required coordinate columns. Found: x={x_col}, y={y_col}, z={z_col}")

    df = _flatten_layer_data(model)

    # Get line_id column name
    line_id_col = model.line_id_column
    if not line_id_col:
        raise ValueError("No line ID column found. Expected one of: title, Line, line, line_id, line_no")

    _compute_xdist(fl, x_col, y_col, line_id_col)
    _compute_sounding_widths(fl, line_id_col)
    cells = _generate_cells(fl, df, x_col, y_col, z_col, line_id_col)
    points_array = _generate_points_array(cells)

    point_coordinates, cell_indices_np, cells_out_vtk, cell_types_out_vtk = _vtk_cell_data(points_array)

    _write_vtk(
        point_coordinates, cell_indices_np, cells_out_vtk, cell_types_out_vtk,
        cells,
        fid,
        attr_out,
        note)

def dump(model, nameorfile, **kw):
    if isinstance(nameorfile, str):
        with open(nameorfile, 'w') as f:
            return _dump(model, f, **kw)
    else:
        return _dump(model, nameorfile, **kw)
