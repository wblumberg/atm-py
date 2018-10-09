import warnings
try:
    from mpl_toolkits.basemap import Basemap as _Basemap
except:
    warnings.warn('There seams to be an issue with importing mpl_toolkits.basemap. Make sure it is installed and working (try: "from mpl_toolkits.basemap import Basemap as _Basemap"). For now plotting on map will not be possible')
import matplotlib.pylab as _plt

colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']

class Site(object):
    def __init__(self, lat = None, lon = None, alt = None, name = None, abbriviation = None, info = None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.name = name
        self.abb = abbriviation
        self.info = info

    def plot(self,
             projection='lcc',
             width=400000 * 7,
             height=500000 * 3,
             abbriviate_name = True,
             resolution='c',
             ax = None):
        if ax:
            a = ax
        else:
            f ,a = _plt.subplots()

        #         self.lat = 36.605700
        #         self.lon = -97.487846
        # self.lat = 78.5
        # self.lon = 18
        # width = 34000
        # height = 22000

        #         width = 400000 * 7
        #         height = 500000 * 3


        bmap = _Basemap  (  # projection='laea',
            projection=projection,
            lat_0=self.lat + 2,
            lon_0=self.lon,
            width=width,
            height=height,
            resolution=resolution,
            ax = a
        )
        # bmap.drawcoastlines()
        bmap.bluemarble()
        bmap.drawcountries(linewidth=2)
        bmap.drawstates()

        # out = bmap.fillcontinents()
        # wcal = np.array([161., 190., 255.]) / 255.
        # boundary = bmap.drawmapboundary(fill_color=wcal)

        fsize = 18
        msize = 8

        lon, lat = self.lon, self.lat # Location Ny-Alesund 78°13′N 15°33′E
        # convert to map projection coords.
        # Note that lon,lat can be scalars, lists or numpy arrays.
        xpt ,ypt = bmap(lon ,lat)
        p, = bmap.plot(xpt ,ypt ,'o')
        p.set_color(colors[1])
        p.set_markersize(msize)

        if abbriviate_name:
            label = self.abb
        else:
            label = self.name
        a.annotate(label, xy=(xpt, ypt),
                   #                 xycoords='data',
                   xytext=(10 ,-10),
                   size = fsize,
                   ha="left",
                   va = 'top',
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=[1 ,1 ,1 ,0.5], ec='black'),
                   )
        return a