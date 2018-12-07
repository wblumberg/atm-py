import warnings
try:
    from mpl_toolkits.basemap import Basemap as _Basemap
except:
    warnings.warn('There seams to be an issue with importing mpl_toolkits.basemap. Make sure it is installed and working (try: "from mpl_toolkits.basemap import Basemap as _Basemap"). For now plotting on map will not be possible')
import matplotlib.pylab as _plt
import os as _os
import numpy as _np
# The following is to ensure that one can use as large of an image as one desires
from PIL import Image as _image
_image.MAX_IMAGE_PIXELS = None

from matplotlib import path as _path


default_colors = _plt.rcParams['axes.prop_cycle'].by_key()['color']



class NetworkStations(object):
    def __init__(self):
        self._stations_list = []

class SubNetworks(object):
    def __init__(self):
        self._network_list = []

    def plot(self, colors = None, **kwargs):
        if isinstance(colors, type(None)):
            site_label_colors = default_colors
        else:
            site_label_colors = colors
        # if isinstance(kwargs['station_symbol_kwargs'], type(None)):
        if ('station_symbol_kwargs' not in kwargs) or (isinstance(kwargs['station_symbol_kwargs'], type(None))):
            kwargs['station_symbol_kwargs'] = {}
        # else:
        #     site_label_colors = colors
        # if 'site_label_marker_size' not in kwargs:
        #     kwargs['site_label_marker_size'] = 8

        # labels = []
        for e,network in enumerate(self._network_list):
            kwargs['station_symbol_kwargs']['color'] = site_label_colors[e]
            a, bmap = network.plot(**kwargs)
            kwargs['bmap'] = bmap
            g = a.get_lines()[-1]
            g.set_label(network.name)
            # label = _plt.Line2D([0], [0], linestyle='', marker='o', label=network.name, markerfacecolor=site_label_colors[e], markersize= )
            # labels.append(label)

        # if 'site_label_font_size' in kwargs:
        #     fontsize = kwargs['site_label_font_size']
        # else:
        #     fontsize = None
        # a.legend(handles=labels, fontsize = fontsize)
        return a, bmap

class Network(object):
    def __init__(self, network_stations, generate_subnetworks = True):
        self.stations = NetworkStations()

        if isinstance(network_stations, list):
            if isinstance(network_stations[0], dict):
                network_stations = network_stations.copy()
                for station in network_stations:
                    station = station.copy()
                    if isinstance(station['abbreviation'], list):
                        station['abbreviation'] = station['abbreviation'][0]
                    # else:
                    #     abb = station['abbreviation']
                    station['name'] = station['name'].replace(' ', '_').replace('.', '')
                    # site = Station(lat=station['lat'],
                    #                lon=station['lon'],
                    #                alt=station['alt'],
                    #                name=name,
                    #                abbreviation=abb,
                    #                info=None)
                    site = Station(**station)
                    self.add_station(site)

            else:
                for station in network_stations:
                    self.add_station(station)

        if generate_subnetworks:
            self._operation_period2sub_network_active()


    def add_station(self, site_instance):
        setattr(self.stations, site_instance.name, site_instance)
        self.stations._stations_list.append(site_instance)

    def add_subnetwork(self, network_instance):
        if not hasattr(self, 'subnetworks'):
            self.subnetworks = SubNetworks()
        setattr(self.subnetworks, network_instance.name, network_instance)
        self.subnetworks._network_list.append(network_instance)

    def plot(self, **kwargs):
        stl = self.stations._stations_list
        # a, bmap = stl[0].plot(plot_only_if_on_map = True, **kwargs)
        # kwargs['bmap'] = bmap
        for e, station in enumerate(stl):
            if 'color' not in kwargs['station_symbol_kwargs']:
                kwargs['station_symbol_kwargs']['color'] = default_colors[1]
            a, bmap = station.plot(plot_only_if_on_map = True, **kwargs)
            kwargs['bmap'] = bmap
        return a,bmap

    def _operation_period2sub_network_active(self):
        has_operation_period = _np.array([hasattr(sta, 'operation_period') for sta in self.stations._stations_list])
        # test if any/all have operation_period
        if not _np.any(has_operation_period):
            return False
        assert (_np.all(has_operation_period))

        # skip if all or none have 'present' in period
        has_present = _np.array(['present' in sta.operation_period for sta in self.stations._stations_list])
        if ((_np.all(has_present)) | (has_present.sum() == 0)):
            return False

        lt = [sta for sta in self.stations._stations_list if 'present' in sta.operation_period]
        active = Network(lt, generate_subnetworks=False)
        active.name = 'active'
        self.add_subnetwork(active)

        lt = [sta for sta in self.stations._stations_list if 'present' not in sta.operation_period]
        inactive = Network(lt, generate_subnetworks=False)
        inactive.name = 'inactive'
        self.add_subnetwork(inactive)


class Station(object):
    def __init__(self, lat = None, lon = None, alt = None, name = None, abbreviation = None, active = None, operation_period = None, info = None, **kwargs):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.name = name
        self.abb = abbreviation
        if info:
            self.info = info
        if active:
            self.active = active
        if operation_period:
            self.operation_period = operation_period
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def plot(self,
             projection='lcc',
             center = 'auto',
             width=400000 * 7,
             height=500000 * 3,
             abbriviate_name = True,
             resolution='c',
             background='blue_marble',
             station_symbol_kwargs = None,
             station_annotation_kwargs = None,
             # site_label_marker_size = 8,
             # site_label_font_size = 18,
             # site_label_color='auto',
             bmap = None,
             plot_only_if_on_map = False,
             ax = None,
             verbose = False):
        """

        Parameters
        ----------
        projection
        width
        height
        abbriviate_name
        resolution
        background: str
            blue_marble: use the blue_marble provided by basemap
            "path_to_file_name": This will use the warpimage function to use the image in the filename ... works with the blue marble stuff (https://visibleearth.nasa.gov/view_cat.php?categoryID=1484)
        plot_only_if_on_map: bool
            as the name says
        ax

        Returns
        -------

        """
        if isinstance(station_symbol_kwargs, type(None)):
            station_symbol_kwargs = {}
        if 'marker' not in station_symbol_kwargs:
            station_symbol_kwargs['marker'] = 'o'
        if 'markersize' not in station_symbol_kwargs:
            station_symbol_kwargs['markersize'] = 8
        if 'color' not in station_symbol_kwargs:
            station_symbol_kwargs['color'] = default_colors[1]

        if isinstance(station_annotation_kwargs, type(None)):
            station_annotation_kwargs = {}
        if 'fontsize' not in station_annotation_kwargs:
            station_annotation_kwargs['size'] = 18

        if bmap:
            a = bmap.ax
        else:
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

            if center == 'auto':
                lat = self.lat
                lon = self.lon
            else:
                lat, lon = center

            bmap = _Basemap  (  # projection='laea',
                projection=projection,
                lat_0=lat,
                lon_0=lon,
                width=width,
                height=height,
                resolution=resolution,
                ax = a
            )
            # bmap.drawcoastlines()
            if background == 'blue_marble':
                bmap.bluemarble()
            elif isinstance(background, type(None)):
                pass
            else:
                assert(_os.path.isfile(background))
                bmap.warpimage(image=background)

            bmap.drawcountries(linewidth=2)
            bmap.drawstates()

        # out = bmap.fillcontinents()
        # wcal = np.array([161., 190., 255.]) / 255.
        # boundary = bmap.drawmapboundary(fill_color=wcal)

        lon, lat = self.lon, self.lat # Location Ny-Alesund 78°13′N 15°33′E
        # convert to map projection coords.
        # Note that lon,lat can be scalars, lists or numpy arrays.

        if plot_only_if_on_map:
            map_bound_path = _path.Path(_np.array([bmap.boundarylons, bmap.boundarylats]).transpose())
            if not map_bound_path.contains_point((lon, lat)):
                if verbose:
                    txt = 'Station {} was not plot, since it is not on the map.'.format(self.name)
                    print(txt)
                return a,bmap

        xpt ,ypt = bmap(lon ,lat)
        p, = bmap.plot(xpt ,ypt ,linestyle = '',**station_symbol_kwargs)
        # if site_label_color == 'auto':
        #     col = colors[1]
        # else:
        #     col = site_label_color

        # p.set_color(station_symbol_kwargs['color'])
        # p.set_markersize()

        if abbriviate_name:
            label = self.abb
        else:
            label = self.name
        a.annotate(label, xy=(xpt, ypt),
                   #                 xycoords='data',
                   xytext=(10 ,-10),
                   size = station_annotation_kwargs['size'],
                   ha="left",
                   va = 'top',
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=[1 ,1 ,1 ,0.5], ec='black'),
                   )
        return a,bmap