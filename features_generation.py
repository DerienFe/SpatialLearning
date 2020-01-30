from load_mol2_data import *
import os
from signature_feature import *

dir = "mol2 structures/"
dirlist = os.listdir(dir)
mol2_dirlist = [item for item in dirlist if item[-5:] == '.mol2']
file_dir = dir + mol2_dirlist[0]

mol = load_atom(file_dir)  # a dictionary contains molecules information and all atom's information

"""compute index map with all atom types from all mol2 files"""
def index_map(dir):
    """

    :param dir: the directory contains all mol2 files
    :return:  the index map
    """
    dirlist = os.listdir(dir)
    mol2_dirlist= [item for item in dirlist if item[-4:] == 'mol2']
    all_atom_types = []
    for file_dir in mol2_dirlist:
        mol = load_atom(dir + file_dir)
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

index_map = index_map(dir)
L = 3
cat_dim = 5
deg_sig = 2
adjacent_matrix = load_adjacent_matrix(file_dir, mol)
"""the feature set on molecule level, output a dictionary of expected signature and log-signature on categorical path
and coordinate path"""
mol_features = mol_features(file_dir, index_map, adjacent_matrix, L, cat_dim, deg_sig)
print(mol_features['expected_cat_sig'].shape)
print(mol_features['expected_cat_logsig'].shape)
print(mol_features['expected_xyz_sig'].shape)
print(mol_features['expected_xyz_logsig'].shape)

""" the feature set on atom level"""
node_index = 1
cat_sig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=0)
cat_logsig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=1)
expected_cat_sig = np.average(cat_sig, 0)
expected_cat_logsig = np.average(cat_logsig, 0)
xyz_sig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=0)
xyz_logsig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=1)
expected_xyz_sig = np.average(xyz_sig, 0)
expected_xyz_logsig = np.average(xyz_logsig, 0)
