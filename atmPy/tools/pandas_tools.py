import numpy as np
import pylab as plt


def ensure_column_exists(df, col_name, col_alt = False):
    """Checks if a particular name is among the column names of a dataframe. Alternative names can be given, which when
    found will be changed to the desired name. The DataFrame will be changed in place. If no matching name is found an
    AttributeError is raised.

    Parameters
    ----------
    col_name: pandas.DataFrame
    col_alt: bool or list
        list of aternative names to look for

    """
    if col_name not in df.columns:
        renamed = False
        if col_alt:
            for dta in col_alt:
                if dta in df.columns:
                    df.rename(columns={dta:col_name}, inplace = True)
                    renamed = True
                else:
                    pass
        if not renamed:
            txt = 'Column %s not found.'%col_name
            if col_alt:
                txt += 'Neither one of the alternatives: %s'(col_alt)
            raise AttributeError(txt)
    return

def plot_dataframe_meshgrid(df, xaxis = 0):
    axes_list = [df.index, df.columns]
    x_index = axes_list[xaxis]

    if xaxis !=0:
        df = df.swapaxes(0,1)
        y_index = axes_list[0]
    else:
        y_index = axes_list[1]

    z = df.values.transpose()
    x = np.repeat(np.array([x_index]), y_index.shape[0], axis = 0)
    y = np.repeat(np.array([y_index]), x_index.shape[0], axis = 0).transpose()

    f,a = plt.subplots()
    pc = a.pcolormesh(x, y , z)


    if 'datetime' in df.index.dtype_str:
        f.autofmt_xdate()
    cb = f.colorbar(pc)
    a.set_xlabel(df.index.name)
    a.set_ylabel(df.columns.name)
    return f,a,pc,cb

def plot_panel_meshgrid(panel, xaxis = 0, yaxis = 1, sub_set = 0, kwargs = {}):

    valid_axes = np.array([0,1,2])
    zaxis = valid_axes[np.logical_and(valid_axes != xaxis, valid_axes != yaxis)][0]
    axes_list = [panel.items, panel.major_axis, panel.minor_axis]

    axes_idx_list_tobe = np.array([int(xaxis),int(yaxis),int(zaxis)])
    axes_idx_list_is = valid_axes.copy()
    x_index = axes_list[xaxis]
    y_index = axes_list[yaxis]
    z_index = axes_list[zaxis]

    if axes_idx_list_tobe[0] != axes_idx_list_is[0]:
        axes_idx_list_is[0], axes_idx_list_is[axes_idx_list_tobe[0]] = axes_idx_list_is[axes_idx_list_tobe[0]], axes_idx_list_is[0]
        panel = panel.swapaxes(0,axes_idx_list_tobe[0])
    if axes_idx_list_tobe[1] != axes_idx_list_is[1]:
        axes_idx_list_is[1], axes_idx_list_is[2] = axes_idx_list_is[2], axes_idx_list_is[1]
        panel = panel.swapaxes(1,2)
    if not np.array_equal(axes_idx_list_is,axes_idx_list_tobe):
        txt = 'not possible'
        raise ValueError(txt)

    z = panel.values[:,:,sub_set].transpose()
    x = np.repeat(np.array([x_index]), y_index.shape[0], axis = 0)
    y = np.repeat(np.array([y_index]), x_index.shape[0], axis = 0).transpose()

    f,a = plt.subplots()
    pc = a.pcolormesh(x, y , z, **kwargs)


    if 'datetime' in panel.items.dtype_str:
        f.autofmt_xdate()
    cb = f.colorbar(pc)
    a.set_xlabel(panel.items.name)
    a.set_ylabel(panel.major_axis.name)
    cb.set_label(panel.minor_axis[sub_set])
    return f,a,pc,cb