# -*- coding: utf-8 -*-
"""
Functions for generic plots
@author: K Houston
"""
import sys
import os

import matplotlib.pyplot as plt

import numpy as np

def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(np.abs(x.astype(np.float64))+1e-20)


def plot_generic(x_data,
        y_data,
        fig = [],
        xy_markers = '-',
        xy_legend = '',
        x_limits=[],
        y_limits=[],
        x_label = '',
        y_label = '',
        fig_title='',
        fig_text_list=[],
        legend_loc = 'upper right',
        display_fig=True,
        savfig_name=''):
    """
    generic plot (single subplot)
    set x_data = x_data1, y_data = y_data1, xy_legend='' or 'legend_label1' for single curve
        x_data = [x_data1,x_data2...], y_data = [y_data1,y_data2,...] , xy_legend='' or ['legend_label1','legend_label2',...] for multiple curves
    """
    if not fig:
        fig = plt.figure(figsize=(10, 6))
        this_is_subplot = False
    else:
        this_is_subplot = True

    if (type(x_data)==list):         # plot multiple curves
        n_curve = len(x_data)
        #print(f'{n_curve = }')
        if (type(xy_markers)==list):
            if (len(xy_markers)==1):
                xy_markers = xy_markers*n_curve
        else:
            xy_markers = [xy_markers]*n_curve
        #print(f'{xy_markers = }')

        do_legend = (len(xy_legend)>0)
        #print(f'{do_legend = }, {xy_legend = }')

        for i_curve in range(n_curve):
            if do_legend:
                plt.plot(x_data[i_curve],y_data[i_curve],xy_markers[i_curve],label=xy_legend[i_curve])
                plt.legend(loc=legend_loc)
            else:
                plt.plot(x_data[i_curve],y_data[i_curve],xy_markers[i_curve])
    else: # plot a single curve
        if len(xy_legend)>0:
            plt.plot(x_data,y_data,xy_markers,label=xy_legend)
            plt.legend(loc=legend_loc)
        else:
            plt.plot(x_data,y_data,xy_markers)
    
    if len(fig_title)>0:
        plt.title(fig_title)
    if len(x_limits)==2:
        plt.xlim(x_limits[0],x_limits[1])
    if len(y_limits)==2:
        plt.ylim(y_limits[0],y_limits[1])
    if len(x_label)>0:
        plt.xlabel(x_label)
    if len(y_label)>0:
        plt.ylabel(y_label)
    for ifig,ft in enumerate(fig_text_list):
        plt.figtext(ft[0],ft[1],ft[2])
    plt.grid()

    if this_is_subplot:
        return

    if len(savfig_name)>0:
        plt.savefig(savfig_name,bbox_inches='tight')
    if display_fig:
        plt.show()
    else:
        plt.close(fig)
    return


