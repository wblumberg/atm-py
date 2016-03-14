from atmPy.aerosols.materials import properties as _properties
from atmPy.tools import pandas_tools as _pandas_tools
import pandas as _pd
from atmPy.general import timeseries as _timeseries
import pdb as _pdb

def zdanovskii_stokes_robinson(data, which = 'refractive_Index'):
    """(Stokes and Robinson,1966)
    Arguments
    ---------
    data: pandas dataframe
        containing chemical composition data
    which: str
        which property to mix ['refractive_Index', 'density', 'kappa_chem']
        """
    materials = _properties.get_commen()
    materials.index = materials.species_name

    essential_elcts = ['ammonium_sulfate',
                                   'ammonium_nitrate',
                                   'ammonium_chloride',
                                   'sodium_chloride',
                                   'sodium_sulfate',
                                   'sodium_nitrate',
                                   'calcium_nitrate',
                                   'calcium_chloride',
                                   'organic_aerosol'
                                  ]

    electrolytes = materials.loc[essential_elcts]
    electrolytes = electrolytes[['refractive_Index', 'density', 'kappa_chem']]

    _pandas_tools.ensure_column_exists(data, 'organic_aerosol', col_alt = ['total_organics'] )

    for e in essential_elcts:
        _pandas_tools.ensure_column_exists(data, e)

    tobemixed = electrolytes[which]
    # _pdb.set_trace()
    numerator = (data * tobemixed / electrolytes.density).sum(axis=1)
    denominator = (data / electrolytes.density).sum(axis=1)
    mixed = numerator/denominator
    df = _pd.DataFrame(mixed, columns=[which])
    ts = _timeseries.TimeSeries(df)
    return ts