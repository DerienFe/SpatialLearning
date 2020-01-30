# PARSING DATABASE
import os
import numpy as np

db_dir = "database/"  # database directory
from molecule_lib import Node
import os


def load_data1(db_dir):
    ligand = []  # will record all the atoms from drug/compound of interest
    protein = []  # will record atoms from protein that are close to the ligand (within 10 Angstrom distance)
    water = []  # will record water atoms (oxygen atoms) that are close to ligand
    print('start')

    for file in os.listdir(db_dir):
        current = open(db_dir + file, "r")
        i=0
        j=0
        k=0
        for row in current:
            line = row.split()

            if row[0:3] == "LGD":  # ligand atoms
                atom_type = line[-1]
                x_y_z = np.asarray(line[3:6], float)
                node1 = Node(atom_type, x_y_z, i)
                ligand.append(node1)
                i = i+1

            if row[0:3] == "PRO":  # protein atoms
                atom_type = line[-1]
                x_y_z = np.asarray(line[3:6], float)
                node1 = Node(atom_type, x_y_z, i)
                protein.append(node1)
                j = j + 1

            if row[0:3] == "HOH":  # water atoms
                atom_type = line[-1]
                x_y_z = np.asarray(line[3:6], float)
                node1 = Node(atom_type, x_y_z, i)
                water.append(node1)
                j = j + 1


    print('end')
    return ligand, protein, water

def get_3D_coordinates(ligand):
    coordinate_data = np.empty((0, 3), float)
    for i in ligand:
        x_y_z = i.features.reshape([1, 3])
        coordinate_data = np.concatenate([coordinate_data, x_y_z], 0)
    return coordinate_data
