import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import dates
from matplotlib import rc
from matplotlib.dates import date2num
from numpy import meshgrid

from atmPy.aerosols.instr.DMA import dma
from atmPy.aerosols.instr.DMA import smps


def proc_scans(*args):

    hagis = smps.SMPS(dma.NoaaWide())

    scan_folder = args[0]#"C:/Users/mrichardson/Documents/HAGIS/SMPS/Scans"
    main_dir = args[1] #'C:/Users/mrichardson/Documents/HAGIS/SMPS/ProcessedData'
    range_low = args[2]
    range_high = args[3]
    darg = args[4]

    # If the directory does not exist, make it...
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    # Change to the directory containing the scan data
    os.chdir(scan_folder)

    # All we care about is the time from the date...
    xfmt = dates.DateFormatter('%H:%M:%S')

    # Begin looping through the dates in the scan directory...
    for i in range(range_low, range_high):
        if i < 10:
            day = "0" + str(i)
        else:
            day = str(i)

        # This will define the year and month through which we scan.
        pattern = darg + day

        get_files(pattern, hagis)

        # This lag is consistent with the the 5 minute scans.
        hagis.lag = 12
        hagis.proc_files()

        # Remove empty values in the lists
        try:
            n_index = hagis.date.index(None)
        except ValueError:
            print("No NONEs found.")
        else:
            print("Index is: " + str(n_index))
            hagis.date = hagis.date[0:n_index]
            hagis.dn_interp = hagis.dn_interp[0:n_index, :]

        xi = date2num(hagis.date)
        XI, YI = meshgrid(xi, hagis.diam_interp)
        Z = hagis.dn_interp.transpose()
        Z[np.where(Z <= 0)] = np.nan
        dataframe = pd.DataFrame(hagis.dn_interp)
        dataframe.index = hagis.date

        # Alot space to store data
        file_name = "hagis_" + pattern
        file = open(main_dir + '/' + file_name + '.csv', 'w')

        dataframe.to_csv(file, mode='a', encoding='utf-8', na_rep="NaN", header=False,
                         line_terminator='\r', index=False)

        dx = np.empty([len(hagis.date)], dtype="str")
        file = open(main_dir + '/' + file_name + '_index.csv', "w+")
        fwriter = csv.writer(file, delimiter='\r')
        dx = [""]*len(hagis.date)
        for e, d in enumerate(hagis.date):
            dx[e] = d.strftime("%m/%d/%y %M:%H:%S")

        fwriter.writerow(dx)
        file.close()

        file = open(main_dir + '/' + 'd.csv', "w+")
        fwriter = csv.writer(file, delimiter='\r')
        fwriter.writerow(hagis.diam_interp)
        file.close()


        np.savetxt(main_dir + '/' + file_name + '_index.csv', dx, fmt="%s")


        pmax = 10**np.ceil(np.log10(np.amax(Z[np.where(Z > 0)])))
        pmin = 1    # 10**np.floor(np.log10(np.amin(Z[np.where(Z > 0)])))

        rc('text', usetex=True)
        rc('font', family='serif')

        fig, ax = plt.subplots()
        pc = ax.pcolor(XI, YI, Z, cmap=plt.cm.jet, norm=colors.LogNorm(pmin, pmax, clip=False), alpha=0.8)

        plt.colorbar(pc)
        plt.yscale('log')
        plt.ylim(5, 1000)
        plt.ylabel(r"\textbf{$D_p$} (nm)")
        plt.xlabel("Time")
        ax.xaxis.set_major_formatter(xfmt)
        plt.title(r"$\frac{dN}{d\log{D_p}}$ for " + pattern)
        fig.autofmt_xdate()
        fig.tight_layout()
        img_name = main_dir + "/Images/" + pattern + ".png"
        #plt.show()
        plt.savefig(img_name)
        print("DAY COMPLETE!!")

    return None

def get_files(pattern, hsmps):
    files = glob.glob("*" + pattern + "*")
    hsmps.files = []
    for i in files:
        hsmps.files.append(open(os.path.abspath(i), 'r'))

