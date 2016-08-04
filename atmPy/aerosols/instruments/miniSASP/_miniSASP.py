from atmPy.general import timeseries
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pylab as plt
from atmPy.tools import plt_tools

class MiniSASP(timeseries.TimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wavelength_channels = [550.4, 460.3, 671.2, 860.7]


        self.__sun_intensities = None

    @property
    def sun_intensities(self, which='short', min_snr=10, moving_max_window=23):
        """ Finds the peaks in all four photo channels (short exposure). It also returns a moving maximum as guide to
         the eye for the more "real" values.

         Parameters
         ----------
         which: 'long' or 'short'.
            If e.g. PhotoA (long) or PhotoAsh(short) are used.
         min_snr: int, optional.
            Minimum signal to noise ratio.
         moving_max_window: in, optionl.
            Window width for the moving maximum.


         Returns
         -------
         TimeSeries instance (AtmPy)
        """
        if not self.__sun_intensities:
            def _moving_max(ds, window=3):
                out = pd.DataFrame(ds, index=ds.index)
                out = pd.rolling_max(out, window)
                out = pd.rolling_mean(out, int(window / 5), center=True)
                return out
            moving_max_window = int(moving_max_window / 2.)
            # till = 10000
            # photos = [self.data.PhotoAsh[:till], self.data.PhotoBsh[:till], self.data.PhotoCsh[:till], self.data.PhotoDsh[:till]]
            if which == 'short':
                photos = [self.data.PhotoAsh, self.data.PhotoBsh, self.data.PhotoCsh, self.data.PhotoDsh]
            elif which == 'long':
                photos = [self.data.PhotoA, self.data.PhotoB, self.data.PhotoC, self.data.PhotoD]
            else:
                raise ValueError('which must be "long" or "short" and not %s' % which)
            # channels = [ 550.4, 460.3, 860.7, 671.2]
            df_list = []
            for e, data in enumerate(photos):
                pos_indx = signal.find_peaks_cwt(data, np.array([10]), min_snr=min_snr)
                out = pd.DataFrame()
                # return out
                # break
                out[str(self.wavelength_channels[e])] = pd.Series(data.values[pos_indx], index=data.index.values[pos_indx])

                out['%s max' % self.wavelength_channels[e]] = _moving_max(out[str(self.wavelength_channels[e])], window=moving_max_window)
                df_list.append(out)

            out = pd.concat(df_list).sort_index()
            out = out.groupby(out.index).mean()
            self.__sun_intensities = Sun_Intensities_TS(out)
        return self.__sun_intensities

    @property
    def optical_depth(self):
        if not self.__optical_depth:





class Sun_Intensities_TS(timeseries.TimeSeries):
    # def plot(self, offset=[0, 0, 0, 0], airmassfct=True, move_max=True, legend=True, all_on_one_axis = False,
    #          additional_axes=False,
    #          errors = False,
    #          rayleigh=True):
    #     """plots ... sorry, but this is a messi function. Things should have been done different, e.g too much data
    #      processing whith the data not put out ... need fixn
    #     Arguments
    #     ---------
    #     offset: list
    #     airmassfct: bool.
    #         If the airmass factor is included or not.
    #         True: naturally the air-mass factor is included in the data, so this does nothing.
    #         False: data is corrected to correct for the slant angle
    #     rayleigh: bool or the aod part of the output of miniSASP.simulate_from_size_dist_LS.
    #         make sure there is no airmassfkt included in this!!
    #     all_on_one_axis: bool or axes instance
    #         if True all is plotted in one axes. If axes instances this axis is used.
    #     """
    #
    #     m_size = 5
    #     m_ewidht = 1.5
    #     l_width = 2
    #     gridspec_kw = {'wspace': 0.05}
    #     no_axes = 4
    #     if all_on_one_axis:
    #         no_axes = 1
    #     if additional_axes:
    #         no_axes = no_axes + additional_axes
    #
    #     if type(all_on_one_axis).__name__ == 'AxesSubplot':
    #         a = all_on_one_axis
    #         f = a.get_figure()
    #     else:
    #         f, a = plt.subplots(1, no_axes, gridspec_kw=gridspec_kw)
    #     columns = ['460.3', '460.3 max', '550.4', '550.4 max', '671.2', '671.2 max', '860.7', '860.7 max']
    #     # peaks_max = [460.3, '460.3 max', 550.4, '550.4 max', 860.7, '860.7 max', 671.2,
    #     #        '671.2 max']
    #     if not all_on_one_axis:
    #         f.set_figwidth(15)
    #     #################
    #     for i in range(int(len(columns) / 2)):
    #         col = plt_tools.wavelength_to_rgb(columns[i * 2]) * 0.8
    #         intens = self.data[columns[i * 2]].dropna()  # .plot(ax = a, style = 'o', label = '%s nm'%colums[i*2])
    #         x = intens.index.get_level_values(1)
    #         if type(rayleigh) == bool:
    #             if rayleigh:
    #                 rayleigh_corr = 0
    #         else:
    #             # print('mach ick')
    #             aodt = rayleigh[float(columns[i * 2])].loc[:, ['rayleigh']]
    #             intenst = intens.copy()
    #             intenst.index = intenst.index.droplevel(['Time', 'Sunelevation'])
    #             aodt_sit = pd.concat([aodt, intenst]).sort_index().interpolate()
    #             aodt_sit = aodt_sit.groupby(aodt_sit.index).mean().reindex(intenst.index)
    #             rayleigh_corr = aodt_sit.rayleigh.values / np.sin(intens.index.get_level_values(2))
    #             # return aodt
    #
    #         if not airmassfct:
    #             amf_corr = np.sin(intens.index.get_level_values(2))
    #         else:
    #             amf_corr = 1
    #         if not all_on_one_axis:
    #             atmp = a[i]
    #         else:
    #             atmp = a
    #
    #
    #         y = (offset[i] - np.log(intens) - rayleigh_corr) * amf_corr
    #         g, = atmp.plot(y, x)
    #         g.set_label('%s nm' % columns[i * 2])
    #         g.set_linestyle('')
    #         g.set_marker('o')
    #         #         g = a.get_lines()[-1]
    #         g.set_markersize(m_size)
    #         g.set_markeredgewidth(m_ewidht)
    #         g.set_markerfacecolor('None')
    #         g.set_markeredgecolor(col)
    #
    #         if move_max:
    #             #             sun_intensities.data.iloc[:,i*2+1].dropna().plot(ax = a)
    #             intens = self.data[
    #                 columns[i * 2 + 1]].dropna()  # .plot(ax = a, style = 'o', label = '%s nm'%colums[i*2])
    #             x = intens.index.values
    #
    #             g, = a[i].plot(offset[i] - np.log(intens), x)
    #             #             g = a.get_lines()[-1]
    #             g.set_color(col)
    #             #             g.set_solid_joinstyle('round')
    #             g.set_linewidth(l_width)
    #             g.set_label(None)
    #
    #         if i != 0 and not all_on_one_axis:
    #             atmp.set_yticklabels([])
    #
    #         if i == 4:
    #             break
    #     if all_on_one_axis:
    #         a.legend()
    #     else:
    #         if legend:
    #             for aa in a:
    #                 aa.legend()
    #     if not airmassfct:
    #         txt = 'OD'
    #     else:
    #         txt = 'OD * (air-mass factor)'`
    #     if all_on_one_axis:
    #         atmp = a
    #     else:
    #         atmp = a[0]
    #     atmp.set_xlabel(txt)
    #     if not all_on_one_axis:
    #         atmp.xaxis.set_label_coords(2.05, -0.07)
    #     atmp.set_ylabel('Altitude (m)')
    #     return a

    def add_sun_elevetion(self, picco):
        """
        doc is not correct!!!

        This function uses telemetry data from the airplain (any timeseries including Lat and Lon) to calculate
        the sun's elevation. Based on the sun's elevation an airmass factor is calculated which the data is corrected for.

        Arguments
        ---------
        sun_intensities: Sun_Intensities_TS instance
        picco: any timeseries instance containing Lat and Lon
        """

        picco_t = timeseries.TimeSeries(picco.data.loc[:, ['Lat', 'Lon', 'Altitude']])  # only Altitude, Lat and Lon
        sun_int_su = self.merge(picco_t)
        out = sun_int_su.get_sun_position()
        #     sun_int_su = sun_int_su.zoom_time(spiral_up_start, spiral_up_end)
        arrays = np.array([sun_int_su.data.index, sun_int_su.data.Altitude, sun_int_su.data.Solar_position_elevation])
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['Time', 'Altitude', 'Sunelevation'])
        sun_int_su.data.index = index
        sun_int_su.data = sun_int_su.data.drop(
            ['Altitude', 'Solar_position_elevation', 'Solar_position_azimuth', 'Lon', 'Lat'], axis=1)
        return sun_int_su
