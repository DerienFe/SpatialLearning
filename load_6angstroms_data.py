import numpy as np
from molecule_lib import Node,Mol
import os

def load_KdKi_data(db_dir):
    """load the atom information includes atom type, coordinates and
    atom index , molecule information includes activity, number of atoms and different atom types
    form 6 angstroms txt file"""
    current = open(db_dir, "r")
    data_file = []
    for row in current:
        line = row.split()
        data_file.append(line)
    activity = float(data_file[2][2])
    LGD_list = [item for item in data_file if item[:1] == ['LGD']]
    PRT_list = [item for item in data_file if item[:1] == ['PRT']]
    water_list = [item for item in data_file if item[:1] == ['HOH']]
    atoms = []
    atom_list = []
    for line in LGD_list:
        atom_type = line[-1]
        x_y_z = np.asarray(line[3:6], float)
        idx = int(line[1])
        node1 = Node(atom_type, x_y_z, idx)
        atoms.append(node1)
        if atom_type not in atom_list:
            atom_list.append(atom_type)
    for line in PRT_list:
        atom_type = line[-1]
        x_y_z = np.asarray(line[4:7], float)
        idx = int(line[1])
        node1 = Node(atom_type, x_y_z, idx)
        atoms.append(node1)
        if atom_type not in atom_list:
            atom_list.append(atom_type)
    for line in water_list:
        atom_type = 'W'
        x_y_z = np.asarray(line[3:6], float)
        idx = int(line[1])
        node1 = Node(atom_type, x_y_z, idx)
        atoms.append(node1)
        if atom_type not in atom_list:
            atom_list.append(atom_type)
    mol_info = Mol(idx, atom_list, activity)
    mol = {'mol_info': mol_info, 'atoms': atoms}
    return mol


def load_KdKi_adj(db_dir,mol):
    current = open(db_dir, "r")
    data_file = []
    for row in current:
        line = row.split()
        data_file.append(line)
    interaction=[item for item in data_file if item[:1] == ['INT']]
    adjacent_matrix = np.zeros([mol['mol_info'].num_atoms, mol['mol_info'].num_atoms])
    for line in interaction:
        adjacent_matrix[int(line[1])-1,int(line[2])-1] = 1
        adjacent_matrix[int(line[2])-1, int(line[1])-1] = 1
    return adjacent_matrix

def index_KdKi_map(dir):
    """

    :param dir: the directory contains all mol2 files
    :return:  the index map
    """
    dirlist = os.listdir(dir)
    mol2_dirlist= [item for item in dirlist if item[-4:] == '.txt']
    all_atom_types = []
    for file_dir in mol2_dirlist:
        mol = load_KdKi_data(dir + file_dir)
        for atom in mol['mol_info'].atom_list:
            if atom not in all_atom_types:
               all_atom_types.append(atom)
    print("atoms include are", all_atom_types)
    index_map = {}
    i = 0
    for atom in sorted(all_atom_types):
        index_map[atom] = i
        i += 1
    return index_map