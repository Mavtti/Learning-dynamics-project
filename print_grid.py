# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:36:09 2019

@author: marc
"""

"""
code taken from Jen Nevens' code : https://github.com/ProjectGameTheory/Plan-Based-Rewards
"""


import matplotlib.pyplot as plt
from ast import literal_eval

def getGrid(fichier):
    grid = []
    with open(fichier, 'rt') as f:
        for line in f:
            grid.insert(0,literal_eval(line[:-1])) #Opposite order
    return grid

def vertical_wall(walls):
    return walls & 1

def horizontal_wall(walls):
    return walls & 2

def printGrid(grid):
    m = len(grid)
    n = len(grid[0])

    ax = plt.gca()
    ax.xaxis.set_ticks(range(1,n+1))
    ax.yaxis.set_ticks(range(1,m+1))
    ax.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])

    ax2 = ax.twinx()
    #ax2.xaxis.set_ticks([float(x)/2 for x in range(2*n)])
    #ax2.yaxis.set_ticks([float(x)/2 for x in range(2*m+1)])
    ax3 = ax.twiny()
    #ax3.xaxis.set_ticks([float(x)/2 for x in range(2*n+1)])

    # create a grid
    ax.grid(True, linestyle='-', linewidth=1)
    # put the grid below other plot elements
    ax.set_axisbelow(True)
    seen_rooms = []
    for i in range(m):
        for j in range(n):
            walls, flag, room = grid[i][j]
            if vertical_wall(walls):
                ax.axvline(x=j,ymin=float(i)/m,ymax=float(i+1)/m,linewidth=4, color='k')
            if horizontal_wall(walls):
                ax.axhline(y=i,xmin=float(j)/n,xmax=float(j+1)/n,linewidth=4, color='k')
            if flag:
                ax.text(float(j+0.5)/n, float(i+0.5)/m, flag, fontsize=12,horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
            if room not in seen_rooms:
                seen_rooms.append(room)
                if room == 'Hall B':
                    ax.text(float(j+0.5)/n, float(i+4.5)/m, room, fontsize=10,
                        weight='bold',
                        verticalalignment='center',
                        transform=ax.transAxes)
                else:
                    ax.text(float(j+0.5)/n, float(i+0.5)/m, room, fontsize=10,
                        weight='bold',
                        verticalalignment='center',
                        transform=ax.transAxes)

    ax.text(float(4.15)/n, float(7+0.5)/m, 'S1', fontsize=10,
                        weight='bold',
                        verticalalignment='center',
                        transform=ax.transAxes)
    ax.axhline(y=m,linewidth=4,color='k')
    ax.axvline(x=n,linewidth=4,color='k')
    ax.text(float(14.20)/n, float(7+0.5)/m, 'S2', fontsize=10,
                        weight='bold',
                        verticalalignment='center',
                        transform=ax.transAxes)
    ax.text(float(0.8)/n, float(1.5)/m, 'Goal', fontsize=8,
                        weight='bold',
                        rotation=-45,
                        verticalalignment='center',
                        transform=ax.transAxes)
    ax.axhline(y=m,linewidth=4,color='k')
    ax.axvline(x=n,linewidth=4,color='k')
    ax.axhline(y=m,linewidth=4,color='k')
    ax.axvline(x=n,linewidth=4,color='k')
    ax2.set_yticklabels([], minor=False)
    ax3.set_xticklabels([],minor=False)
    ax2.tick_params(axis=u'both', which=u'both',length=0)
    ax3.tick_params(axis=u'both', which=u'both',length=0)
    ax2.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    plt.draw()
    plt.show()