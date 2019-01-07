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
            grid.append(literal_eval(line[:-1])) #Opposite order
    return grid

grid = getGrid("grid.txt")
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
    ax3 = ax.twiny()

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


    ax.text(float(4.15)/n, float(7.5)/m, 'S1', fontsize=10, weight='bold', verticalalignment='center', transform=ax.transAxes)
    ax.text(float(14.15)/n, float(7.5)/m, 'S2', fontsize=10, weight='bold', verticalalignment='center', transform=ax.transAxes)
    ax.text(float(0.9)/n, float(1.5)/m, 'Goal', fontsize=8, weight='bold', rotation=-45, verticalalignment='center', transform=ax.transAxes)

    ax.axhline(y=m,linewidth=4,color='k')
    ax.axvline(x=n,linewidth=4,color='k')
    plt.draw()
    plt.show()