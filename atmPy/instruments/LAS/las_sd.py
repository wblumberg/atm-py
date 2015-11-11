import pandas as pd
import tkinter as tk
from tkinter import filedialog as fd
import numpy as np

class LAS(object):
    def __init__(self):
        self.las_pd = None
        self.diam = None
        self.hk = None

        self.sd = None

        self.bincenters = None

        self.ismass = True
        self.dndlogdp = None

    def read_data(self, fname=''):
        if fname == '':
            _gui = tk.Tk()
            file = fd.askopenfiles()
            _gui.destroy()

            self.las_pd = pd.read_csv(file[0], header=None, sep='\t')

        self.diam = self.las_pd.iloc[0:2, 15:-1]

        self.sd = self.las_pd.iloc[3:-1, 15:-1]
        self.hk = self.las_pd.iloc[3:-1, 0:14]

        self.hk.columns = self.las_pd.iloc[0, 0:14].values

        self.bincenters = 10 ** ((np.log10(self.diam.iloc[0, :].values) +
                                  np.log10(self.diam.iloc[1, :].values))/2)

        dlogdp = np.log10(self.diam.iloc[1, :].values)-np.log10(self.diam.iloc[0, :].values)
        print(dlogdp)

        self.sd.columns = self.bincenters

        # Make sure these are floats...
        self.hk.iloc[:, 2:14] = self.hk.iloc[:, 2:14].astype(np.float)

        # Retrieves size distribution per scc
        self.sd = (self.sd.transpose()/(self.hk.Sample.values*self.hk['Accum.'].values)).transpose()*60

        self.dndlogdp = self.sd.copy()/dlogdp

        # Convert pressure to millibars
        self.hk['Pres.'] *= 10

    def convertSD(self):

        convert = 273.15/1013.25*self.hk['Pres.']/self.hk.Box

        if self.ismass:
            print('Converting from standard.')

            self.sd = (self.sd.transpose()/convert).transpose()
            self.dndlogdp = (self.dndlogdp.transpose()/convert).transpose()
            self.ismass = False
        else:
            print('Converting to standard.')

            self.sd = (self.sd.transpose()*convert).transpose()
            self.dndlogdp = (self.dndlogdp.transpose()*convert).transpose()
            self.ismass = True









