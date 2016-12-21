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
        self.assertLess(abs((out.relative_humidity.data.index.values - pd.to_datetime(soll.index).values).sum() / np.timedelta64(1, 's')), 1e-10)
        # self.assertTrue(np.all(out.relative_humidity.data.index.values == pd.to_datetime(soll.index).values))

        ## rest
        soll.columns.name = out.relative_humidity.data.columns.name
        # self.assertTrue(np.all(out.relative_humidity.data.values == soll.values))
        self.assertLess(abs((out.relative_humidity.data.values - soll.values).sum()), 1e-2)
        # temp
        soll = pd.read_csv(test_data_folder + '1twr10xC1_temp.csv', index_col=0,
                           dtype={'temp_25m': np.float32, 'temp_60m': np.float32}
                           )
        soll.columns.name = out.temperature.data.columns.name
        # self.assertTrue(np.all(out.temperature.data.values == soll.values))
        self.assertLess(abs((out.temperature.data.values - soll.values).sum()), 1e-2)

        # vapor pressure
        soll = pd.read_csv(test_data_folder + '1twr10xC1_p_vapor.csv', index_col=0,
                           dtype={'vap_pres_25m': np.float32, 'vap_pres_60m': np.float32}
                           )
        soll.columns.name = out.vapor_pressure.data.columns.name
        # self.assertTrue(np.all(out.vapor_pressure.data.values == soll.values))
        self.assertLess(abs((out.vapor_pressure.data.values - soll.values).sum()), 1e-2)


class SizeDistTest(TestCase):
    def test_concentrations(self):
        sd = size_distribution.sizedistribution.simulate_sizedistribution(diameter=[15, 3000],
                                                                          numberOfDiameters=50,
                                                                          centerOfAerosolMode=222,
                                                                          widthOfAerosolMode=0.18,
                                                                          numberOfParticsInMode=888)

        self.assertEqual(round(sd.particle_number_concentration, 4) , round(888.0, 4))
        self.assertEqual(round(float(sd.particle_surface_concentration.values), 4) , round(194.42186363605904, 4))
        self.assertEqual(round(float(sd.particle_volume_concentration.values), 4) , round(11.068545094055812, 4))

        sd.properties.particle_density = 2.2
        self.assertEqual(round(float(sd.particle_mass_concentration), 4), round(24.350799206922783, 4))


    def test_moment_conversion(self):
        sd = size_distribution.sizedistribution.simulate_sizedistribution(diameter=[15, 3000],
                                                                          numberOfDiameters=50,
                                                                          centerOfAerosolMode=222,
                                                                          widthOfAerosolMode=0.18,
                                                                          numberOfParticsInMode=888)

        sd_dNdDp = sd.convert2dNdDp()
        sd_dNdlogDp = sd.convert2dNdlogDp()
        sd_dSdDp = sd.convert2dSdDp()
        sd_dSdlogDp = sd.convert2dSdlogDp()
        sd_dVdDp = sd.convert2dVdDp()
        sd_dVdlogDp = sd.convert2dVdlogDp()

        folder = test_data_folder

        # sd.save_csv(folder + 'aerosols_size_dist_moments_sd.nc')
        # sd_dNdDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dNdDp.nc')
        # sd_dNdlogDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dNdlogDp.nc')
        # sd_dSdDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dSdDp.nc')
        # sd_dSdlogDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dSdlogDp.nc')
        # sd_dVdDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dVdDp.nc')
        # sd_dVdlogDp.save_csv(folder + 'aerosols_size_dist_moments_sd_dVdlogDp.nc')

        sd_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd.nc')
        sd_dNdDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dNdDp.nc')
        sd_dNdlogDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dNdlogDp.nc')
        sd_dSdDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dSdDp.nc')
        sd_dSdlogDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dSdlogDp.nc')
        sd_dVdDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dVdDp.nc')
        sd_dVdlogDp_soll = size_distribution.sizedistribution.read_csv(folder + 'aerosols_size_dist_moments_sd_dVdlogDp.nc')

        threshold = 1e-10
        msg = '\nthreshold: {}\nisnan: {}\nisnotnan: {}'.format((sd.data.values.max() * threshold),
                                                 np.isnan(sd.data.values - sd_soll.data.values).sum(),
                                                 (~np.isnan(sd.data.values - sd_soll.data.values)).sum())
        self.assertLess(abs((sd.data.values - sd_soll.data.values)).sum() , (sd.data.values.max() * threshold), msg = msg)
        self.assertLess(abs((sd_dNdDp.data.values - sd_dNdDp_soll.data.values)).sum() , (sd_dNdDp.data.values.max() * threshold))
        self.assertLess(abs((sd_dSdDp.data.values - sd_dSdDp_soll.data.values)).sum() , (sd_dSdDp.data.values.max() * threshold))
        self.assertLess(abs((sd_dVdDp.data.values - sd_dVdDp_soll.data.values)).sum() , (sd_dVdDp.data.values.max() * threshold))
        self.assertLess(abs((sd_dNdlogDp.data.values - sd_dNdlogDp_soll.data.values)).sum() , (sd_dNdlogDp.data.values.max() * threshold))
        self.assertLess(abs((sd_dSdlogDp.data.values - sd_dSdlogDp_soll.data.values)).sum() , (sd_dSdlogDp.data.values.max() * threshold))
        self.assertLess(abs((sd_dVdlogDp.data.values - sd_dVdlogDp_soll.data.values)).sum() , (sd_dVdlogDp.data.values.max() * threshold))

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

        sd.optical_properties_settings.refractive_index = 1.56
        sd.optical_properties_settings.wavelength = 515

        fname = os.path.join(test_data_folder, 'aerosols_size_dist_LS_optprop.nc')
        sdl = atmPy.read_file.netCDF(fname)

        # self.assertTrue(np.all(sd.optical_properties.aerosol_optical_depth_cumulative_VP.data.values == sdl.data.values))
        self.assertLess(abs((sd.optical_properties.aerosol_optical_depth_cumulative_VP.data.values - sdl.data.values).sum()), 1e-10)