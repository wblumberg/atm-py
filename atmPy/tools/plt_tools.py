from matplotlib.colors import LinearSegmentedColormap
import numpy as np
###

blue = np.array([0.,13.,120.])/255.
orange = np.array([255.,102.,0.])/255.
green = np.array([9.,84.,0.])/255.
purple = np.array([138.,0.,84.])/255.
yellow = np.array([184.,159.,0.])/255.
turques = np.array([0.,204.,214.])/255.
red = np.array([120.,17.,0.])/255.

color_cycle = [blue,orange,green,purple,yellow,turques,red]



def setRCParams_fontsize(plt,style= 'paper'):
    textsizescale_paper = 2.
    textsizescale_talk = 2.5
    if style == 'paper':
        plt.rcParams['font.size'] = plt.rcParams['figure.figsize'][0]*textsizescale_paper 
    if style == 'talk':
        plt.rcParams['font.size'] = plt.rcParams['figure.figsize'][0]*textsizescale_talk 

    return
        
def setRcParams(plt, style = 'paper'):
    ''' style: paper,talk'''
    
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.figsize'] = [10.0, 8.0]   #talk: [6.0, 5.0]
    setRCParams_fontsize(plt,style = style)
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['figure.autolayout'] =  True
    plt.rcParams['text.usetex'] = False
#    plt.rcParams['mathtext.rm'] =  'sans' 
#    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.default']='regular'
  

def get_colorMap_intensity():
    """ according to the colorweel intensity II"""
    color1 = [0.0,4./255,76./255] 
    color2 = [49./255., 130./255., 0.0]
    color3 = [1.,197./255.,98./255.]
    color4 = [245./255., 179./255., 223./255.]
    color5 = [ 216./255., 1.0,1.0]
    cdict = {'red':   ((0.0, color1[0], color1[0]),
                       (0.25,color2[0] ,color2[0]),
                       (0.5,color3[0] ,color3[0]),
                       (0.75,color4[0] ,color4[0]),
                       (1.00,color5[0] ,color5[0])),
    
             'green': ((0.0, color1[1], color1[1]),
                       (0.25,color2[1] , color2[1]),
                       (0.5,color3[1] ,color3[1]),
                       (0.75,color4[1] ,color4[1]),
                       (1.0,color5[1] ,color5[1])),
    
             'blue':  ((0.0, color1[2], color1[2]),
                       (0.25, color2[2], color2[2]),
                       (0.5, color3[2] ,color3[2]),
                       (0.75,color4[2] ,color4[2]),
                       (1.0,color5[2] ,color5[2]))
            }
    
    hag_cmap  = LinearSegmentedColormap('hag_cmap',cdict)
    hag_cmap.set_bad('black')
    return hag_cmap
    
def get_colorMap_intensity_r():
    """ according to the colorweel intensity II"""
    color5 = [0.0, 4./255, 76./255]
    color4 = [49./255., 130./255., 0.0]
    color3 = [1.,197./255., 98./255.]
    color2 = [245./255., 179./255., 223./255.]
    color1 = [216./255., 1.0, 1.0]
    cdict = {'red':   ((0.0, color1[0], color1[0]),
                       (0.25,color2[0] ,color2[0]),
                       (0.5,color3[0] ,color3[0]),
                       (0.75,color4[0] ,color4[0]),
                       (1.00,color5[0] ,color5[0])),
    
             'green': ((0.0, color1[1], color1[1]),
                       (0.25,color2[1] , color2[1]),
                       (0.5,color3[1] ,color3[1]),
                       (0.75,color4[1] ,color4[1]),
                       (1.0,color5[1] ,color5[1])),
    
             'blue':  ((0.0, color1[2], color1[2]),
                       (0.25, color2[2], color2[2]),
                       (0.5, color3[2] ,color3[2]),
                       (0.75,color4[2] ,color4[2]),
                       (1.0,color5[2] ,color5[2]))
            }
    
    hag_cmap = LinearSegmentedColormap('hag_cmap',cdict)
    hag_cmap.set_bad('black')
    return hag_cmap

def get_colorMap_heat():
    """ according to the colorweel heat"""
    color1 = np.array([0.0,14.,161.])/255.
    color2 = np.array([0., 125., 11.])/255.
    color3 = np.array([255.,255.,255.])/255.
    color4 = np.array([255., 172., 0.])/255.
#    color5 = np.array([ 184., 0.,18.])/255.
    color5 = np.array([ 163., 0.,119.])/255.
    cdict = {'red':   ((0.0, color1[0], color1[0]),
                       (0.25,color2[0] ,color2[0]),
                       (0.5,color3[0] ,color3[0]),
                       (0.75,color4[0] ,color4[0]),
                       (1.00,color5[0] ,color5[0])),
    
             'green': ((0.0, color1[1], color1[1]),
                       (0.25,color2[1] , color2[1]),
                       (0.5,color3[1] ,color3[1]),
                       (0.75,color4[1] ,color4[1]),
                       (1.0,color5[1] ,color5[1])),
    
             'blue':  ((0.0, color1[2], color1[2]),
                       (0.25, color2[2], color2[2]),
                       (0.5, color3[2] ,color3[2]),
                       (0.75,color4[2] ,color4[2]),
                       (1.0,color5[2] ,color5[2]))
            }
    
    hag_cmap  = LinearSegmentedColormap('hag_cmap',cdict)
    hag_cmap.set_bad('black')
    return hag_cmap
    

    