import pandas as pd
import os

def get_commen():
    this_dir = os.path.split(__file__)[0]
    fname = os.path.join(this_dir, 'materials.xlsx')
    mat_df = pd.read_excel(fname)
    return mat_df

