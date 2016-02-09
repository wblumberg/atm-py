import pandas as pd


def various():
    fname = 'materials.xlsx'
    mat_df = pd.read_excel(fname, index_col=0)
    return mat_df

