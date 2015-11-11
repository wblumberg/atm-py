


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