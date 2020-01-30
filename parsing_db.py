#PARSING DATABASE
import os
import numpy as np

from molecule_lib import Node
from load_data import *

db_dir = "database/" # database directory
ligand, protein, water = load_data1(db_dir)
coordinate_data = get_3D_coordinates(ligand)
print(coordinate_data)


from mpl_toolkits import mplot3d
#%matplotlib inline
import matplotlib.pyplot as plt


color_map = {'C': 'r',
             'O': 'g',
             'N': 'y',
             'S': 'b',
             'P': 'm'}

print(color_map['C'])

xdata = coordinate_data[1:, 0]
ydata = coordinate_data[1:, 1]
zdata = coordinate_data[1:, 2]

L = np.shape(xdata)[0]

#import networkx, random
#adjacent_matrix =  np.random.randint(2, size=(L, L))
#s = np.random.binomial(L, 0.005, L*L)
#print(s)
#adjacent_matrix = s.reshape((L,L))
#print('ajacent_matrix = {}'.format(adjacent_matrix))

adjacent_matrix = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if (np.abs(j-i) <3):
            adjacent_matrix[i, j] = 1


fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(L):
    for j in range(L):
        if adjacent_matrix[i,j] == 1:
            ax.plot(xdata[[i,j]], ydata[[i,j]], zdata[[i,j]],  color = 'b')

for i in range(L):
    atom_t = ligand[i].ntype
    ax.scatter3D(xdata[i], ydata[i], zdata[i], s = 50,  color = color_map[atom_t])
    ax.text(xdata[i]+0.2, ydata[i], zdata[i], atom_t, style='italic')
ax.grid(False)
plt.show()


"""
# The necessary data (atoms and positions for each component) is in the three lists
# You can decide how you want to write this data out in a format suitable for you.
"""
