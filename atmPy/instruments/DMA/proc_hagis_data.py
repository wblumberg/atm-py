from atmPy.instruments.DMA import dma
from atmPy.instruments.DMA import smps
#import atmPy.sizedistribution as sd
from matplotlib import colors
import matplotlib.pyplot as plt
from numpy import meshgrid
import numpy as np
from matplotlib.dates import date2num
from matplotlib import dates
import pandas as pd
import os
import glob
from matplotlib import rc
import csv

def proc_scans(*args, **kwargs):

    hagis = smps.SMPS(dma.NoaaWide())

    scan_folder = "C:/Users/mrichardson/Documents/HAGIS/SMPS/Scans"
    main_dir = 'C:/Users/mrichardson/Documents/HAGIS/SMPS/ProcessedData'

    # If the directory does not exist, make it...
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    # Change to the directory containing the scan data
    os.chdir(scan_folder)

    # All we care about is the time from the date...
    xfmt = dates.DateFormatter('%H:%M:%S')

    # Begin looping through the dates in the scan directory...
    for i in range(15, 16):
        if i < 10:
            day = "0" + str(i)
        else:
            day = str(i)

        # This will define the year and month through which we scan.
        pattern = "201503" + day

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
       #file_index = open(main_dir + '/' + file_name + '_index.csv', 'w')

        dataframe.to_csv(file, mode='a', encoding='utf-8', na_rep="NaN", header=False,
                         line_terminator='\r', index=False)

        dx = np.empty([len(hagis.date)], dtype="str")
        file = open(main_dir + '/' + file_name + '_index.csv', "w+")
        fwriter = csv.writer(file)
        dx = [""]*len(hagis.date)
        for e, d in enumerate(hagis.date):
            print(d.strftime("%m/%d/%y %M:%H:%S"))
            dx[e] = d.strftime("%m/%d/%y %M:%H:%S")

        fwriter.writerow(dx)

        print(dx)
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
        img_name = "C:/Users/mrichardson/Documents/HAGIS/SMPS/images/" + pattern + ".png"
        plt.show()
        plt.savefig(img_name)

    return None

def get_files(pattern, hsmps):
    files = glob.glob("*" + pattern + "*")
    hsmps.files = []
    for i in files:
        hsmps.files.append(open(os.path.abspath(i), 'r'))

