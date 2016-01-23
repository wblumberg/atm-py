# -*- coding: utf-8 -*-
"""
This module is out of date!

@author: htelg
"""
import pandas as pd

from atmPy.aerosols.size_distr import sizedistribution
from atmPy.for_removal.tools import diameter_binning


def load_PMEL_APS(fname):
    na_values = [u'StartDateTime', u'Dp_1', u'Dp_2', u'Dp_3', u'Dp_4', u'Dp_5', u'Dp_6', u'Dp_7', u'Dp_8', u'Dp_9', u'Dp_10', u'Dp_11', u'Dp_12', u'Dp_13', u'Dp_14', u'Dp_15', u'Dp_16', u'Dp_17', u'Dp_18', u'Dp_19', u'Dp_20', u'Dp_21', u'Dp_22', u'Dp_23', u'Dp_24', u'Dp_25', u'Dp_26', u'Dp_27', u'Dp_28', u'Dp_29', u'Dp_30', u'Dp_31', u'Dp_32', u'Dp_33', u'Dp_34', u'Dp_35', u'Dp_36', u'Dp_37', u'Dp_38', u'Dp_39', u'Dp_40', u'Dp_41', u'Dp_42', u'Dp_43', u'Dp_44', u'Dp_45', u'Dp_46', u'Dp_47', u'Dp_48', u'Dp_49', u'Dp_50', u'Dp_51', u'Dp_52',u'dNdlogDp_1', u'dNdlogDp_2', u'dNdlogDp_3', u'dNdlogDp_4', u'dNdlogDp_5', u'dNdlogDp_6', u'dNdlogDp_7', u'dNdlogDp_8', u'dNdlogDp_9', u'dNdlogDp_10', u'dNdlogDp_11', u'dNdlogDp_12', u'dNdlogDp_13', u'dNdlogDp_14', u'dNdlogDp_15', u'dNdlogDp_16', u'dNdlogDp_17', u'dNdlogDp_18', u'dNdlogDp_19', u'dNdlogDp_20', u'dNdlogDp_21', u'dNdlogDp_22', u'dNdlogDp_23', u'dNdlogDp_24', u'dNdlogDp_25', u'dNdlogDp_26', u'dNdlogDp_27', u'dNdlogDp_28', u'dNdlogDp_29', u'dNdlogDp_30', u'dNdlogDp_31', u'dNdlogDp_32', u'dNdlogDp_33', u'dNdlogDp_34', u'dNdlogDp_35', u'dNdlogDp_36', u'dNdlogDp_37', u'dNdlogDp_38', u'dNdlogDp_39', u'dNdlogDp_40', u'dNdlogDp_41', u'dNdlogDp_42', u'dNdlogDp_43', u'dNdlogDp_44', u'dNdlogDp_45', u'dNdlogDp_46', u'dNdlogDp_47', u'dNdlogDp_48', u'dNdlogDp_49', u'dNdlogDp_50', u'dNdlogDp_51', u'dNdlogDp_52']
    tab = pd.read_csv(fname, sep = '\t', na_values=na_values)
    tab = tab.dropna()
    newIndex = pd.to_datetime(tab.StartDateTime.values)
    tab.index = newIndex
    reducedTab = tab.iloc[:,53:]
    bincenters = tab.iloc[0,1:53].values*1000
    binedges,newColnames = diameter_binning.bincenters2binsANDnames(bincenters)
    reducedTab.columns = newColnames
    dist = sizedistribution.aerosolSizeDistribution(reducedTab, binedges, 'dNdlogDp')
    return dist