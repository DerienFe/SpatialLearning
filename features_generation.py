from load_mol2_data import *
import os
from signature_feature import *
dir = "mol2 structures/"
dirlist = os.listdir(dir)
file_dir=dir + dirlist[0]

mol=load_atom(file_dir)# a dictionary contains molecules information and all atom's information

"""run a simple code to find all atom types"""
def find_all_atom_types(dir):
    dirlist = os.listdir(dir)
    all_atom_types = []
    for file_dir in dirlist:
       if file_dir!='.DS_Store':
          mol=load_atom(dir+file_dir)

          for atom in mol['mol_info'].atom_list:
              if atom not in all_atom_types:
                 all_atom_types.append(atom)
    return(all_atom_types)
all_atom_types=find_all_atom_types(dir)
print("atoms include are",all_atom_types)
index_map = {'C': 0,
             'O': 1,
             'H': 2,
             'N': 3,
             'S': 4}
L=3
cat_dim=5
deg_sig=2
adjacent_matrix=load_adjacent_matrix(file_dir,mol)
"""the feauture set on molecule level, output a dictionary of expected signature and logsignature on categorical path
and coordinate path"""
mol_features=mol_features(file_dir,index_map,adjacent_matrix,L,cat_dim,deg_sig)
print(mol_features['expected_cat_sig'].shape)
print(mol_features['expected_cat_logsig'].shape)
print(mol_features['expected_xyz_sig'].shape)
print(mol_features['expected_xyz_logsig'].shape)

""" the feaure set on atom level"""
node_index=1
cat_sig = categorical_path_sig(node_index,mol,adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=0)
cat_logsig = categorical_path_sig(node_index,mol, adjacent_matrix, index_map, L,cat_dim, deg_sig,flag= 1)
expected_cat_sig=np.average(cat_sig,0)
expected_cat_logsig = np.average(cat_logsig, 0)
xyz_sig = coordinate_path_sig(node_index,mol, adjacent_matrix, L, deg_sig, flag=0)
xyz_logsig = coordinate_path_sig(node_index, mol,adjacent_matrix, L, deg_sig, flag=1)
expected_xyz_sig = np.average(xyz_sig, 0)
expected_xyz_logsig = np.average(xyz_logsig, 0)




