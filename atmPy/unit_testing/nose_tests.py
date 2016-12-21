from unittest import TestCase
import numpy as np
import pandas as pd
import os
test_data_folder = os.path.join(os.path.dirname(__file__), 'test_data/')
# print(test_data_folder)
# test_data_folder = './test_data/'
#### data archives
######## ARM
from atmPy.data_archives.arm import _read_data
import atmPy
from atmPy.aerosols import size_distribution

class ArmDataTests(TestCase):
    def test_1twr10xC1(self):
        out = _read_data.read_cdf(test_data_folder, data_product='1twr10xC1')
        out = out['1twr10xC1']

        # rh
        soll = pd.read_csv(test_data_folder + '1twr10xC1_rh.csv', index_col=0,
                           dtype={'rh_25m': np.float32, 'rh_60m': np.float32}
                           )

        ## index
        self.assertLess((out.relative_humidity.data.index.values - pd.to_datetime(soll.index).values).sum() / np.timedelta64(1, 's'), 1)
        # self.assertTrue(np.all(out.relative_humidity.data.index.values == pd.to_datetime(soll.index).values))

        ## rest
        soll.columns.name = out.relative_humidity.data.columns.name
        self.assertTrue(np.all(out.relative_humidity.data.values == soll.values))

        # temp
        soll = pd.read_csv(test_data_folder + '1twr10xC1_temp.csv', index_col=0,
                           dtype={'temp_25m': np.float32, 'temp_60m': np.float32}
                           )
        soll.columns.name = out.temperature.data.columns.name
        self.assertTrue(np.all(out.temperature.data.values == soll.values))

        # vapor pressure
        soll = pd.read_csv(test_data_folder + '1twr10xC1_p_vapor.csv', index_col=0,
                           dtype={'vap_pres_25m': np.float32, 'vap_pres_60m': np.float32}
                           )
        soll.columns.name = out.vapor_pressure.data.columns.name
        self.assertTrue(np.all(out.vapor_pressure.data.values == soll.values))


class SizeDistTest(TestCase):
    def test_opt_prop_LS(self):
        sd = size_distribution.sizedistribution.simulate_sizedistribution_layerseries(diameter=[10, 2500],
                                                                                      numberOfDiameters=100,
                                                                                      heightlimits=[0, 6000],
                                                                                      noOflayers=100,
                                                                                      layerHeight=[500.0, 4000.0],
                                                                                      layerThickness=[100.0, 300.0],
                                                                                      layerDensity=[1000.0, 50.0],
                                                                                      layerModecenter=[200.0, 800.0],
                                                                                      widthOfAerosolMode=0.2)

        sd.optical_properties_settings.refractive_index = 1.5
        sd.optical_properties_settings.wavelength = 550

        fname = os.path.join(test_data_folder, 'aerosols_size_dist_LS_optprop.nc')
        sdl = atmPy.read_file.netCDF(fname)

        self.assertTrue(np.all(sd.optical_properties.aerosol_optical_depth_cumulative_VP.data.values == sdl.data.values))