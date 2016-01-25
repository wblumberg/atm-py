from hagpack.projects.arm import _tools
import pandas as pd
from atmPy import timeseries
import numpy as np

info = """base_time
Base time in Epoch
()
--------
time_offset
Time offset from base_time
(1440,)
--------
time
Time offset from midnight
(1440,)
--------
qc_time
Quality check results on field: Time offset from midnight
(1440,)
--------
flags_CMDL
AOS system status
(1440, 4)
--------
N_CPC_1
Condensation nuclei concentration number
(1440,)
--------
N_CCN_1
Cloud droplet concentration number from summation of condensation nuclei counter size bins
(1440,)
--------
Ba_G_Dry_10um_PSAP1W_1
Aerosol light absorption coefficient, green channel, 10 um particle diameter
(1440,)
--------
Bs_B_Dry_10um_Neph3W_1
Aerosol total light scattering coefficient, reference nephelometer blue channel, 10 um particle diameter
(1440,)
--------
Bs_G_Dry_10um_Neph3W_1
Aerosol total light scattering coefficient, reference nephelometer green channel, 10 um particle diameter
(1440,)
--------
Bs_R_Dry_10um_Neph3W_1
Aerosol total light scattering coefficient, reference nephelometer red channel, 10 um particle diameter
(1440,)
--------
Bbs_B_Dry_10um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, reference nephelometer blue channel, 10 um particle diameter
(1440,)
--------
Bbs_G_Dry_10um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, reference nephelometer green channel, 10 um particle diameter
(1440,)
--------
Bbs_R_Dry_10um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, reference nephelometer red channel, 10 um particle diameter
(1440,)
--------
Bs_B_Wet_10um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer blue channel, 10 um particle diameter
(1440,)
--------
Bs_G_Wet_10um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer green channel, 10 um particle diameter
(1440,)
--------
Bs_R_Wet_10um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer red channel, 10 um particle diameter
(1440,)
--------
Bbs_B_Wet_10um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer blue channel, 10 um particle diameter
(1440,)
--------
Bbs_G_Wet_10um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer green channel, 10 um particle diameter
(1440,)
--------
Bbs_R_Wet_10um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer red channel, 10 um particle diameter
(1440,)
--------
Ba_G_Dry_1um_PSAP1W_1
Aerosol light absorption coefficient, green channel, 1 um particle diameter
(1440,)
--------
Bs_B_Dry_1um_Neph3W_1
Aerosol total light scattering coefficient, ref. humidograph nephelometer blue channel, 1 um particle diameter
(1440,)
--------
Bs_G_Dry_1um_Neph3W_1
Aerosol total light scattering coefficient, ref. humidograph nephelometer green channel, 1 um particle diameter
(1440,)
--------
Bs_R_Dry_1um_Neph3W_1
Aerosol total light scattering coefficient, ref. humidograph nephelometer red channel, 1 um particle diameter
(1440,)
--------
Bbs_B_Dry_1um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, ref. humidograph nephelometer blue channel, 1 um particle diameter
(1440,)
--------
Bbs_G_Dry_1um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, ref. humidograph nephelometer green channel, 1 um particle diameter
(1440,)
--------
Bbs_R_Dry_1um_Neph3W_1
Aerosol backwards-hemispheric light scattering coefficient, ref. humidograph nephelometer red channel, 1 um particle diameter
(1440,)
--------
Bs_B_Wet_1um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer blue channel, 1 um particle diameter
(1440,)
--------
Bs_G_Wet_1um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer green channel, 1 um particle diameter
(1440,)
--------
Bs_R_Wet_1um_Neph3W_2
Aerosol total light scattering coefficient, humidograph nephelometer red channel, 1 um particle diameter
(1440,)
--------
Bbs_B_Wet_1um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer blue channel, 1 um particle diameter
(1440,)
--------
Bbs_G_Wet_1um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer green channel, 1 um particle diameter
(1440,)
--------
Bbs_R_Wet_1um_Neph3W_2
Aerosol backwards-hemispheric light scattering coefficient, humidograph nephelometer red channel, 1 um particle diameter
(1440,)
--------
RH_MainInlet
Relative humidity at inlet
(1440,)
--------
T_MainInlet
Temperature at inlet
(1440,)
--------
RH_NephInlet_Dry
Relative humidity at reference nephelometer inlet
(1440,)
--------
T_NephInlet_Dry
Temperature at reference nephelometer inlet
(1440,)
--------
RH_NephVol_Dry
Relative humidity inside reference nephelometer
(1440,)
--------
T_NephVol_Dry
Temperature inside reference nephelometer
(1440,)
--------
RH_preHG
Relative humidity of humidograph preheater
(1440,)
--------
T_preHG
Temperature of humidograph preheater
(1440,)
--------
RH_postHG
Relative humidity of humidograph controller
(1440,)
--------
T_postHG
Temperature of humidograph controller
(1440,)
--------
RH_NephInlet_Wet
Relative humidity inside at wet nephelometer inlet
(1440,)
--------
T_NephInlet_Wet
Temperature at wet nephelometer inlet
(1440,)
--------
RH_NephVol_Wet
Relative humidity inside of wet nephelometer
(1440,)
--------
T_NephVol_Wet
Temperature inside of wet nephelometer
(1440,)
--------
RH_Ambient
Relative humidity in ambient air
(1440,)
--------
T_Ambient
Temperature in ambient air
(1440,)
--------
P_Ambient
Ambient pressure
(1440,)
--------
P_Neph_Dry
Pressure inside reference nephelometer
(1440,)
--------
P_Neph_Wet
Pressure inside wet nephelometer
(1440,)
--------
WindSpeed
Wind speed
(1440,)
--------
WindDirection
Wind Direction, relative to true North
(1440,)
--------
Ba_B_Dry_10um_PSAP3W_1
Absorption coefficient blue, corrected for flow and sample area, 3 wavelength PSAP, 10 um size cut
(1440,)
--------
Ba_G_Dry_10um_PSAP3W_1
Absorption coefficient green, corrected for flow and sample area, 3 wavelength PSAP, 10 um size cut
(1440,)
--------
Ba_R_Dry_10um_PSAP3W_1
Absorption coefficient red, corrected for flow and sample area, 3 wavelength PSAP, 10 um size cut
(1440,)
--------
Ba_B_Dry_1um_PSAP3W_1
Absorption coefficient blue, corrected for flow and sample area, 3 wavelength PSAP, 1 um size cut
(1440,)
--------
Ba_G_Dry_1um_PSAP3W_1
Absorption coefficient green, corrected for flow and sample area, 3 wavelength PSAP, 1 um size cut
(1440,)
--------
Ba_R_Dry_1um_PSAP3W_1
Absorption coefficient red, corrected for flow and sample area, 3 wavelength PSAP, 1 um size cut
(1440,)
--------
lat
North latitude
()
--------
lon
East longitude
()
--------
alt
Altitude above mean sea level
()
"""



def _parse_netCDF(file_obj):

    abs_coeff = ['Ba_G_Dry_10um_PSAP1W_1',
                'Ba_G_Dry_1um_PSAP1W_1',
                'Ba_B_Dry_10um_PSAP3W_1',
                'Ba_G_Dry_10um_PSAP3W_1',
                'Ba_R_Dry_10um_PSAP3W_1',
                'Ba_B_Dry_1um_PSAP3W_1',
                'Ba_G_Dry_1um_PSAP3W_1',
                'Ba_R_Dry_1um_PSAP3W_1',
                ]

    scat_coeff =   ['Bs_B_Dry_10um_Neph3W_1',
                        'Bs_G_Dry_10um_Neph3W_1',
                        'Bs_R_Dry_10um_Neph3W_1',
                        'Bs_B_Wet_10um_Neph3W_2',
                        'Bs_G_Wet_10um_Neph3W_2',
                        'Bs_R_Wet_10um_Neph3W_2',
                        'Bs_B_Dry_1um_Neph3W_1',
                        'Bs_G_Dry_1um_Neph3W_1',
                        'Bs_R_Dry_1um_Neph3W_1',
                        'Bs_B_Wet_1um_Neph3W_2',
                        'Bs_G_Wet_1um_Neph3W_2',
                        'Bs_R_Wet_1um_Neph3W_2',
                        ]


    bscat_coeff_vars = ['Bbs_B_Dry_10um_Neph3W_1',
                        'Bbs_G_Dry_10um_Neph3W_1',
                        'Bbs_R_Dry_10um_Neph3W_1',
                        'Bbs_B_Wet_10um_Neph3W_2',
                        'Bbs_G_Wet_10um_Neph3W_2',
                        'Bbs_R_Wet_10um_Neph3W_2',
                        'Bbs_B_Dry_1um_Neph3W_1',
                        'Bbs_G_Dry_1um_Neph3W_1',
                        'Bbs_R_Dry_1um_Neph3W_1',
                        'Bbs_B_Wet_1um_Neph3W_2',
                        'Bbs_G_Wet_1um_Neph3W_2',
                        'Bbs_R_Wet_1um_Neph3W_2',
                        ]

    RH = ['RH_NephVol_Dry',
          'RH_NephVol_Wet']

    def var2ts(file_obj, var_list, index, column_name):
        """extracts the list of variables from the file_obj and puts them all in one data frame"""
        df = pd.DataFrame(index = index)
        for var in var_list:
            variable = file_obj.variables[var]
            data = variable[:]
            fill_value = variable.missing_data
            data = np.ma.masked_where(data == fill_value, data)
            df[var] = pd.Series(data, index = index)
        df.index.name = 'Time'
        df.columns.name = column_name
        out = timeseries.TimeSeries(df)
        return out

    index = _tools._get_time(file_obj)


    out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
    out['abs_coeff'] = var2ts(file_obj, abs_coeff, index, 'abs_coeff_1/Mm')
    out['scatt_coeff'] = var2ts(file_obj, scat_coeff, index, 'scatt_coeff_1/Mm')
    out['back_scatt'] = var2ts(file_obj, bscat_coeff_vars, index, 'back_scatt_1/Mm')
    out['RH'] = var2ts(file_obj, RH, index, 'RH')

    out.info = info
    return out


def _concat_rules(files):
    out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
    out['abs_coeff'] = timeseries.TimeSeries(pd.concat([i['abs_coeff'].data for i in files]))
    out['scatt_coeff'] = timeseries.TimeSeries(pd.concat([i['scatt_coeff'].data for i in files]))
    out['back_scatt'] = timeseries.TimeSeries(pd.concat([i['back_scatt'].data for i in files]))
    out['RH'] = timeseries.TimeSeries(pd.concat([i['RH'].data for i in files]))
    return out