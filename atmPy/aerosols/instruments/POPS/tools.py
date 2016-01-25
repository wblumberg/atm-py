import numpy as np
import pylab as plt
import sys

def arc_length(r, d):
    """ length of the arc of a circle segment (given by a secant) as a function
    of the circle radius r and the shortest distance (d) between the center of the
    circle and the (center) of the secant
    """ 

    try:
        r[r<d] = d
    except TypeError:
        pass        
    
    l_arc = 2 * np.arccos(float(d) / r) * r
    return l_arc
    
def arc_length_alpha(r, alpha):
    """ length of the arc of a circle segment (given by a secant) as a function
    of the circle radius r and the angle which defines the segment.
    """ 
    l_arc = alpha * r
    return l_arc
    
def segment_hight(r,cordLength):
    """hight of a circle segment as a function of circle radius (r) and chord length (cordLength)."""
    r = float(r)
    cordLength = float(cordLength)
    h = r - np.sqrt( r**2 - (cordLength**2 / 4) )
    return h
    
def segment_angle(r, arcLength):
    """angle in rad of a segment as a funktion of the radius and the arc length
    http://en.wikipedia.org/wiki/Circular_segment"""

    theta = arcLength/r
    return theta
    
def sphereSegment_radius(R, theta):
    """Radius of the segment of a sphere as a function of the spheres radius R 
    and the angle. See drawing: sphereSegment_radius.svg.
    theta: rad
    """
    
    r = R * np.cos(theta)
    
    return r

def alphamax_fromGeometry(h,dm):
    """ maximum angle to the normal vector at the center of the mirror at which 
    light is still collected by the mirror. Note in general this angle has to be 
    considered on all sides of the normal vector!
    h: distance from particle jet to plain which is defined by the top of the mirror
    dm: diameter of the mirror"""
    
    alphaMax = np.arctan(float(dm) / (2 * float(h)))
    
    return alphaMax
    
def sphereRadius_fromGeometry(h,dm):
    """ This function defines the radius (r) of the sphere which we are cutting 
    and slicing in order to get the intensities expacted to be scattered to the 
    PMT. 
    h: distance from particle jet to plain which is defined by the top of the mirror
    dm: diameter of the mirror
    """
    r = np.sqrt((float(dm)**2 / 4) + float(h)**2)
    
    return r

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx,array[idx]

def find_angleRange(center, aMax, radiantArray, test = False):
    """ Returns two arrays of all angles and there indeces between "center-aMax" and "center + aMax" in the array radiantArray.
    center: center angle in rad
    aMax: maximum angle from center in rad
    radiantArray: ordered radiant values"""
    center = float(center)
    aMax = float(aMax)
    startIdx, startvalue = find_nearest(radiantArray, center - aMax)
    endIdx, endvalue = find_nearest(radiantArray, center + aMax)
    centerIdx, centervalue = find_nearest(radiantArray, center)
    indexArray = np.arange(startIdx,endIdx+1)
    if test:
        print('start', startIdx, startvalue)
        print('end', endIdx, endvalue)
        print('center', centerIdx, centervalue)
        print('radiantArray', radiantArray)
        print('len(radiantArray[startIdx:endIdx+1])',len(radiantArray[startIdx:endIdx+1]))
        print('radiantArray[startIdx:endIdx+1]', radiantArray[startIdx:endIdx+1])
        print('indexArray', indexArray)
        print('len(indexArray)', len(indexArray))
    return radiantArray[startIdx:endIdx+1], indexArray
#     return 

def refIndex_polystyrene(waveLength):
    """Refractive index of polystyrene as a function of wavelength (um). 
    Ref: http://refractiveindex.info/?group=PLASTICS&material=PS"""
    if waveLength < .350:
        raise ValueError("""For wavelengths below 350 nm polystyrene has a extingtion 
                        coefficient >0. Can't use this function unless you implement absorption!""")
    x = float(waveLength)
    n = np.sqrt( 2.610025 - 6.143673e-2 * x**2 - 1.312267e-1 * x**-2 + 6.865432e-2 * x**-4 - 1.295968e-2 * x**-6 + 9.055861e-4 * x**-8 )
    return n


def gauss_function(x, x0, fwhm, norm = 1):
    """ x: float or array
        x0: center
        fwhm: ...
        norm: this is the value the integral is normalized to. This might be usefull, when different lightsources with different intensities are used
        """
        
    if np.all(x > 10) or np.all(x0 >10) or np.all(fwhm > 0.05):
#     if True in np.greater(x, 10) or True in np.greater(x0, 10) or True in np.greater(w,0.1)
        answer = raw_input("""This is probably an error, are you sure you are giving the wavelength in um?!? (y)""")
        if answer.lower() == 'n':
            sys.exit('as you wish ... exit!')
    w = float(fwhm)/(2*np.sqrt(2*np.log(2)))
    a = float(norm) / (w * np.sqrt(2 * np.pi))
    X = (x - x0)/ w
    gsf = a * np.exp(-.5 * X**2)
    return gsf

def test_plot():
    x = np.linspace(.390,.420,50)
    y = gauss_function(x,.405,0.005)
    np.savetxt('output/gaussFkt.dat',np.array([x,y]).transpose())
    plt.plot(x,y)
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
#     print arc_length(1, 0)
#     print sphereSegment_radius(10, np.deg2rad(0))
#     print alphamax_fromGeometry(10, 20)
#     print sphereRadius_fromGeometry(1e10,10)
     print(find_angleRange(np.pi/2.,.5,np.linspace(0,2 * np.pi,20), test=True))
#     print segment_hight(20,0)
#     print refIndex_polystyrene(.4358)
#    test_plot()