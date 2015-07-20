__author__ = 'htelg'

import numpy as np
from numpy import linalg as la


def cart2spheric(xyz):
    """
    Returns
    -------
    (r,theta, phi):
        r: length
        theta: polar angle (between vector and z axesl)
        rho: azimuth angle (in x,y plane)
    """
    #     sph = np.zeros(xyz.shape)#np.hstack((xyz, np.zeros(xyz.shape)))
    #     xy = xyz[:,0]**2 + xyz[:,1]**2
    #     sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
    #     sph[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #     #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #     sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    if len(xyz.shape) == 1:
        xyz = np.array([xyz])
        d1 = True
    else:
        d1 = False
    rtp = np.zeros(xyz.shape)  # np.hstack((xyz, np.zeros(xyz.shape)))
    rtp[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    rtp[:, 1] = np.arccos(xyz[:, 2] / rtp[:, 0])
    rtp[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    if d1:
        return rtp[0]
    else:
        return rtp


def spheric2cart(rtp):
    """r,theta,phi"""
    if len(rtp.shape) == 1:
        rtp = np.array([rtp])
        d1 = True
    else:
        d1 = False
    xyz = np.zeros(rtp.shape)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    if d1:
        return xyz[0]
    else:
        return xyz


def angleBetweenVectors(a, b):
    """returns the angle between two vectors
    Arguments
    ---------
    a,b: np.array of shape (n,3,) or (3,)

    Returns
    -------
    """
    if a.shape != b.shape:
        raise ValueError('a and be must have the same shape')
    if len(a.shape) == 1:
        return (np.arccos(np.dot(a, b) / (la.norm(a) * la.norm(b))))

    else:
        angles = np.zeros(a.shape[0])
        for e, i in enumerate(a):
            angles[e] = (np.arccos(np.dot(a[e], b[e]) / (la.norm(a[e]) * la.norm(b[e]))))
        return angles
