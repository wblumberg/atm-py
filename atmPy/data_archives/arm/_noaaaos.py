from atmPy.general import timeseries
# from atmPy.data_archives.arm import _tools
import pandas as pd
from atmPy.data_archives.arm._netCDF import ArmDataset

import pdb as _pdb

class ArmDatasetSub(ArmDataset):
    def __init__(self,*args, **kwargs):
        super(ArmDatasetSub,self).__init__(*args, **kwargs)
        self._parse_netCDF()

    def _parse_netCDF(self):

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

        RH_neph = ['RH_NephVol_Dry',
              'RH_NephVol_Wet']

        def var2ts(self, var_list, column_name):
            """extracts the list of variables from the file_obj and puts them all in one data frame"""
            df = pd.DataFrame(index = self.time_stamps)
            for var in var_list:
                data = self._read_variable(var)
                # variable = file_obj.variables[var]
                # data = variable[:]
                # fill_value = variable.missing_data
                # data = np.ma.masked_where(data == fill_value, data)
                df[var] = pd.Series(data, index = self.time_stamps)
            df.columns.name = column_name
            out = timeseries.TimeSeries(df)
            return out

        # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
        # out['abs_coeff'] = var2ts(self, abs_coeff, 'abs_coeff_1/Mm')
        # out['scatt_coeff'] = var2ts(self, scat_coeff, 'scatt_coeff_1/Mm')
        # out['back_scatt'] = var2ts(self, bscat_coeff_vars, 'back_scatt_1/Mm')
        # out['RH'] = var2ts(self, RH, 'RH')
        # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
        self.abs_coeff = var2ts(self, abs_coeff, 'abs_coeff_1/Mm')
        self.scatt_coeff = var2ts(self, scat_coeff, 'scatt_coeff_1/Mm')
        self.back_scatt = var2ts(self, bscat_coeff_vars, 'back_scatt_1/Mm')
        self.RH_nephelometer = var2ts(self, RH_neph, 'RH')
        # _pdb.set_trace()
        # df = pd.DataFrame(self.read_variable('RH_interDMA'), index = self.time_stamps, columns=size_bins)
        # df.columns.name = 'size_bin_center_nm'
        # self.RH_interDMA = timeseries.TimeSeries(df)
        # self.RH_ambient = self.read_variable(timeseries.TimeSeries('RH_Ambient'))
        # self.temperature_ambient = self.read_variable('T_Ambient')


    def plot_all(self):
        self.abs_coeff.plot()
        self.back_scatt.plot()
        self.scatt_coeff.plot()
        self.RH_nephelometer.plot()



def _concat_rules(arm_data_objs):
    out = ArmDatasetSub(False)
    out.abs_coeff = timeseries.TimeSeries(pd.concat([i.abs_coeff.data for i in arm_data_objs]))
    out.back_scatt = timeseries.TimeSeries(pd.concat([i.back_scatt.data for i in arm_data_objs]))
    out.scatt_coeff = timeseries.TimeSeries(pd.concat([i.scatt_coeff.data for i in arm_data_objs]))
    out.RH_nephelometer = timeseries.TimeSeries(pd.concat([i.RH_nephelometer.data for i in arm_data_objs]))
    # out.RH_ambient = timeseries.TimeSeries(pd.concat([i.RH_ambient.data for i in arm_data_objs]))
    # out.temperature_ambient = timeseries.TimeSeries(pd.concat([i.temperature_ambient.data for i in arm_data_objs]))
    out.time_stamps = out.abs_coeff.data.index
    # out = _tools.ArmDict(plottable= ['abs_coeff', 'scatt_coeff', 'back_scatt'] )
    # out['abs_coeff'] = timeseries.TimeSeries(pd.concat([i['abs_coeff'].data for i in files]))
    # out['scatt_coeff'] = timeseries.TimeSeries(pd.concat([i['scatt_coeff'].data for i in files]))
    # out['back_scatt'] = timeseries.TimeSeries(pd.concat([i['back_scatt'].data for i in files]))
    # out['RH'] = timeseries.TimeSeries(pd.concat([i['RH'].data for i in files]))
    return out