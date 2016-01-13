from atmPy.aerosols.instr.DMA import dma
from atmPy.aerosols.instr.DMA import smps
#import atmPy.sizedistribution as sd
from matplotlib import colors
import matplotlib.pyplot as plt
from numpy import meshgrid
import numpy as np
from matplotlib.dates import date2num
from matplotlib import dates
import pandas as pd
import os
import sys

def proc_scans(*args, **kwargs):
    try:
        hagis = smps.SMPS(dma.NoaaWide())

        scan_folder = "C:/Users/mrichardson/Documents/HAGIS/SMPS/Scans"

        os.chdir(scan_folder)





        hagis.lag = 10

        hagis.proc_files()

        try:
            n_index = hagis.date.index(None)
        except ValueError:
            print("No NONEs found.")
        else:
            print("Index is: " + str(n_index))
            hagis.date = hagis.date[0:n_index]
            hagis.dn_interp = hagis.dn_interp[0:n_index, :]

        xfmt = dates.DateFormatter('%m/%d %H:%M')

        xi = date2num(hagis.date)
        XI, YI = meshgrid(xi, hagis.diam_interp)
        Z = hagis.dn_interp.transpose()
        Z[np.where(Z <= 0)] = np.nan
        dataframe = pd.DataFrame(hagis.dn_interp)
        dataframe.index = hagis.date

        main_dir = 'C:/Users/mrichardson/Documents/HAGIS/SMPS/ProcessedData'

        if not os.path.exists(main_dir):
            os.mkdir(main_dir)

        file_name = input('File name: ')

        file = open(main_dir + '/' + file_name + '.csv', 'w')

        file_index = open(main_dir + '/' + file_name + '_index.csv', 'w')

        dataframe.to_csv(file, mode='a', encoding='utf-8', na_rep="NaN", header=False,
                         line_terminator='\r', index=False)

        dataframe.to_csv(file_index, mode='a', encoding='utf-8', na_rep="NaN", header=False,
                         line_terminator='\r', index=True, columns=[])

        #datewriter = csv.writer(file_index, delimiter=' ')
        #for i in hagis.date:
        #    datewriter.writerow(i.strftime('%Y/%m/%d %H:%M:%S'))

        pmax = 1e6  # 10**np.ceil(np.log10(np.amax(Z[np.where(Z > 0)])))
        pmin = 1    # 10**np.floor(np.log10(np.amin(Z[np.where(Z > 0)])))
        fig, ax = plt.subplots()
        pc = ax.pcolor(XI, YI, Z, cmap=plt.cm.jet, norm=colors.LogNorm(pmin, pmax, clip=False), alpha=0.8)

        plt.colorbar(pc)
        plt.yscale('log')
        plt.ylim(5, 1000)
        ax.xaxis.set_major_formatter(xfmt)
        fig.autofmt_xdate()
        fig.tight_layout()

        plt.show()
    except:
        print("Unexpected error in test_smps_class:", sys.exc_info()[0])
        print('whoops!')

    return hagis
