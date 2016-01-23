#!/opt/local/bin/python2.7
# -*- coding: utf-8 -*-

'''
POPS peak data processing GUI

author: Hagen Telg
last modified: September 2011
'''

import os

import matplotlib
import numpy as np
import wx
from atmPy.instruments.POPS import peaks

from atmPy.for_removal.POPS import calibration

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

wildcard_peak = "POPS peak file (*Peak.bin)|*Peak.bin|" \
                "All files (*.*)|*.*"
wildcard_cal = "POPS peak file (*.peak)|*.csv|" \
               "All files (*.*)|*.*"


def do_atm_stuff(mainframe, paths_cal, paths_peaks):
    print(paths_cal, paths_peaks)
    cal = calibration.read_csv(paths_cal[0])
    #    out = cal.plot_calibration()
    #    out[0].show()
    m = peaks.read_binary(paths_peaks)
    m.apply_calibration(cal)
    dist = m.peak2numberdistribution(bins=np.logspace(np.log10(150), np.log10(2500), 100))

    #    pname,fname = os.path.split(paths_peaks[0])
    dist_name = paths_peaks[0] + '_dist.csv'
    dist.save_csv(dist_name)
    msg = """Size distribution was saved at
%s.""" % dist_name
    mainframe.report_success(msg)
    f, a, pc, cb = dist.plot()
    frame = CanvasFrame(f)
    frame.Show(True)
    return


class CanvasFrame(wx.Frame):
    def __init__(self, f):
        wx.Frame.__init__(self, None, -1, 'size distribution', size=(550, 350))
        self.SetBackgroundColour(wx.NamedColor('WHITE'))
        self.figure = f
        #        self.axes = self.figure.add_subplot(111)
        #        t = arange(0.0, 3.0, 0.01)
        #        s = sin(2 * pi * t)
        #        self.axes.plot(t, s)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizerAndFit(self.sizer)
        self.add_toolbar()

    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if wx.Platform == '__WXMAC__':
            self.SetToolBar(self.toolbar)
        else:
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.update()

    def OnPaint(self, event):
        self.canvas.draw()


class Example(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(Example, self).__init__(*args, **kwargs)

        self.InitUI()

    def InitUI(self):

        menubar = wx.MenuBar()

        fileMenu = wx.Menu()

        #        fileMenu.Append(wx.ID_NEW, '&New')


        opn = wx.MenuItem(fileMenu, wx.ID_OPEN, '&Open')
        fileMenu.AppendItem(opn)
        self.Bind(wx.EVT_MENU, self.OnOpen, opn)

        fileMenu.Append(wx.ID_SAVE, '&Save')
        fileMenu.AppendSeparator()

        qmi = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+W')
        fileMenu.AppendItem(qmi)

        self.Bind(wx.EVT_MENU, self.OnQuit, qmi)

        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)

        toolbar1 = wx.ToolBar(self)
        tsize = (24, 24)
        open_bmp = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, tsize)

        open_t = toolbar1.AddLabelTool(wx.ID_ANY, '', open_bmp)
        self.Bind(wx.EVT_TOOL, self.OnOpen, open_t)


        #        toolbar1.AddLabelTool(wx.ID_ANY, '', wx.Bitmap('topen.png'))
        #        toolbar1.AddLabelTool(wx.ID_ANY, '', wx.Bitmap('tsave.png'))
        toolbar1.Realize()

        self.SetSize((350, 80))
        self.SetTitle('POPS_peak2SD')
        self.Centre()
        self.Show(True)

    def OnOpen(self, e):
        #        self.log.WriteText("CWD: %s\n" % os.getcwd())

        # Create the dialog. In this case the current directory is forced as the starting
        # directory for the dialog, and no default file name is forced. This can easilly
        # be changed in your program. This is an 'open' dialog, and allows multitple
        # file selections as well.
        #
        # Finally, if the directory is changed in the process of getting files, this
        # dialog is set up to change the current working directory to the path chosen.
        dlg = wx.FileDialog(
            self, message="Choose a calibration file",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=wildcard_cal,
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
        )

        # Show the dialog and retrieve the user response. If it is the OK response, 
        # process the data.
        if dlg.ShowModal() == wx.ID_OK:
            # This returns a Python list of files that were selected.
            paths_cal = dlg.GetPaths()

            for path in paths_cal:
                print(path)

        dlg = wx.FileDialog(
            self, message="Choose peak file(s)",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=wildcard_peak,
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
        )

        # Show the dialog and retrieve the user response. If it is the OK response, 
        # process the data.
        if dlg.ShowModal() == wx.ID_OK:
            # This returns a Python list of files that were selected.
            paths_peaks = dlg.GetPaths()

            for path in paths_peaks:
                print(path)

        do_atm_stuff(self, paths_cal, paths_peaks)
        # Compare this with the debug above; did we change working dirs?

        # Destroy the dialog. Don't do this until you are done with it!
        # BAD things can happen otherwise!
        dlg.Destroy()

    def OnQuit(self, e):
        self.Close()

    def report_success(self, msg):
        dlg = wx.MessageDialog(self, msg,
                               'Message Box',
                               wx.OK | wx.ICON_INFORMATION
                               # wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                               )
        dlg.ShowModal()
        dlg.Destroy()


def main():
    ex = wx.App()
    Example(None)
    ex.MainLoop()


if __name__ == '__main__':
    main()
