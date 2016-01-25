import pandas as pd

def _get_time(file_obj):
    bt = file_obj.variables['base_time']
    toff = file_obj.variables['time_offset']
    time = pd.to_datetime(0) + pd.to_timedelta(bt[:].flatten()[0], unit = 's') + pd.to_timedelta(toff[:], unit = 's')
    return time

class ArmDict(dict):
    def __init__(self, plottable = [], plot_kwargs = {}, *args):
        super(ArmDict,self).__init__(self,*args)
        self.plottable = plottable
        self.plot_kwargs = plot_kwargs

    def plot(self, which = 'all', fig_size = None):
        if which == 'all':
            for item in self.plottable:
                out = self[item].plot(**self.plot_kwargs)
                if type(out).__name__ == 'AxesSubplot':
                    a = out
                elif len(out) == 4:
                    a = out[1]

                a.set_title(item)
                if fig_size:
                    f = a.get_figure()
                    f.set_size_inches((fig_size))
                # return out