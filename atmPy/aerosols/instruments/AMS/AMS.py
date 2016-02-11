from atmPy.general import timeseries
from atmPy.aerosols.materials import properties
import numpy as np
import pandas as pd

def ion2electrolyte_mass_concentration(ion_concentrations, ions, electrolytes):
    cct = ion_concentrations.drop(['total_organics'])/ions.molecular_weight

    ions['molar_concentration'] = cct

    cations = ions.loc[ions['ion'] == 'cat']
    anions = ions.loc[ions['ion'] == 'an']

    n_cat = cations.molar_concentration
    z_cat = cations.charge_on_ion
    z_an = anions.molar_concentration
    n_an = anions.charge_on_ion

    eps_cats = z_cat * n_cat / (z_cat * n_cat).sum()
    eps_ans = z_an * n_an / (z_an * n_an).sum()

    ###### material_elct['molar_concentration'] = np.nan

    for elect in electrolytes.index:
#         print(elect)
        elect_prop = electrolytes.loc[elect]

        eps_an = eps_ans.loc[elect_prop.loc['anion']]
        eps_cat = eps_cats.loc[elect_prop.loc['cation']]
        M_cat = ions.loc[elect_prop.loc['cation']].loc['molecular_weight']
        M_an = ions.loc[elect_prop.loc['anion']].loc['molecular_weight']
        n_an = ions.loc[elect_prop.loc['anion']].loc['molar_concentration']
        n_cat = ions.loc[elect_prop.loc['cation']].loc['molar_concentration']
        M_elect = elect_prop.loc['molecular_weight']

        n_elect = ((eps_cat * n_an * M_an) + (eps_an * n_cat * M_cat)) / M_elect

        electrolytes.loc[elect,'molar_concentration'] = n_elect

    electrolytes['mass_concentration'] = electrolytes.molar_concentration * electrolytes.molecular_weight

    electrolytes.drop(['molar_concentration'], axis=1, inplace=True)

    return electrolytes

class AMS_Timeseries(timeseries.TimeSeries):

    def calculate_electrolyte_mass_concentrations(self):
        # ion_mass_concentration = self.data
        materials = properties.get_commen()

        materials.index = materials.species_name
        material_ions = materials.loc[['ammonium','sulfate', 'nitrate', 'chloride', 'sodium', 'calcium' ]]
        material_ions = material_ions.dropna(axis=1)
        material_ions = material_ions.drop(['Species', 'species_name'], axis = 1)

        cats = material_ions.loc[material_ions['ion'] == 'cat']
        ans = material_ions.loc[material_ions['ion'] == 'an']

        material_elct = materials.loc[['ammonium_sulfate',
                                       'ammonium_nitrate',
                                       'ammonium_chloride',
                                       'sodium_chloride',
                                       'sodium_sulfate',
                                       'sodium_nitrate',
                                       'calcium_nitrate',
                                       'calcium_chloride'
                                      ]]
        material_elct = material_elct.dropna(axis=1)

        np.zeros((self.data.shape[0],material_ions.shape[0]))
        df = pd.DataFrame(columns=material_elct.index, index = self.data.index)
        for i in self.data.index:
        #     print(i)
            cct = self.data.loc[i]
            electro = ion2electrolyte_mass_concentration(cct, material_ions, material_elct)
            df.loc[i] = electro.mass_concentration
        df['total_organics'] = self.data.total_organics
        return timeseries.TimeSeries(df)