from load_mol2_data import *
import os
from signature_feature import *

"""load mol2 data"""
'''
dir = "mol2 structures/"
dirlist = os.listdir(dir)
mol2_dirlist = [item for item in dirlist if item[-5:] == '.mol2']
file_dir = dir + mol2_dirlist[0]

mol = load_atom(file_dir)  # a dictionary contains molecules information and all atom's information
'''
"""compute index map with all atom types from all mol2 files"""

'''
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
'''
""" the feature set on atom level"""
'''
node_index = 1
cat_sig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=0)
cat_logsig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=1)
expected_cat_sig = np.average(cat_sig, 0)
expected_cat_logsig = np.average(cat_logsig, 0)
xyz_sig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=0)
xyz_logsig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=1)
expected_xyz_sig = np.average(xyz_sig, 0)
expected_xyz_logsig = np.average(xyz_logsig, 0)

mol_features = mol_KdKi_features(file_dir, index_map, L, cat_dim, deg_sig,'sig')
print(mol_features['expected_cat_sig'].shape)
print(mol_features['expected_cat_logsig'].shape)
print(mol_features['expected_xyz_sig'].shape)
print(mol_features['expected_xyz_logsig'].shape)
'''

"""load Kd data"""
'''
from load_6angstroms_data import *
dir = "6 Angstroms/database_Kd_6A/"
dirlist = os.listdir(dir)
dirlist = [item for item in dirlist if item[-4:] == '.txt']
file_dir = dir + dirlist[0]

mol = load_KdKi_data(file_dir)  # a dictionary contains molecules information and all atom's information
index_map = index_KdKi_map(dir)
L = 2
cat_dim =len(index_map)
deg_sig = 2
adj = load_KdKi_adj(file_dir, mol)
"""the feature set on molecule level, output a dictionary of expected signature and log-signature on categorical path
and coordinate path"""
features=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol_features = mol_KdKi_features(file_dir, index_map, L, cat_dim, deg_sig, 'sig')
    features.append(np.concatenate([mol_features['expected_cat_sig'],mol_features['expected_xyz_sig']],axis=1))
'''
'''
dir = "6 Angstroms/database_Kd_6A/"
dirlist = os.listdir(dir)
dirlist = [item for item in dirlist if item[-4:] == '.txt']
adj_list=[]
len_list=[]
activity=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol = load_KdKi_ligand_data(file_dir)
    len_list.append(mol['mol_info'].num_atoms)
    adj_list.append(load_KdKi_adj(file_dir, mol))
    activity.append(mol['mol_info'].activity)



features_mat,adj_mat=padding_zeros(features,adj_list,len_list)

np.save('features.npy',features_mat)
np.save('adj.npy',adj_mat)
np.save('length.npy',np.array(len_list))
np.save('activity.npy',activity)
features_1=np.load(features.npy)


'''


'''compute original features, xyz data and one hot encoded atom type information'''
from load_6angstroms_data import *
dir = "database_10angstroms/"
dirlist = os.listdir(dir)
dirlist = [item for item in dirlist if item[-4:] == '.txt']


index_map = index_KdKi_map(dir)
from sklearn import preprocessing
le = preprocessing.OneHotEncoder()
le.fit(np.array(list(index_map.keys())).reshape([10,1]))
original_features=[]
adj_list=[]
len_list=[]
activity=[]
num_atoms=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol = load_KdKi_data(file_dir)
    len_list.append(mol['mol_info'].num_atoms)
    adj_list.append(load_KdKi_ligand_adj(file_dir, mol))
    activity.append(mol['mol_info'].activity)
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol = load_KdKi_data(file_dir)
    temp=[]
    for j in range(mol['mol_info'].num_atoms):
        temp.append(np.concatenate([mol['atoms'][j].features,le.transform([[mol['atoms'][j].ntype]]).toarray().flatten]))
    original_features.append(np.array(temp))


def padding_zeros(feature_list,adj_list,len_list):
    max_len=max(len_list)
    pad_feature_list=[]
    pad_adj_list=[]
    for i in range(len(len_list)):
        print(i)
        pad_features = np.zeros([max_len, feature_list[0].shape[1]])
        pad_adj = np.zeros([max_len, max_len])
        pad_features[:len_list[i],:]=feature_list[i]
        pad_feature_list.append(pad_features)
        pad_adj[:len_list[i],:len_list[i]]=adj_list[i]
        pad_adj_list.append(pad_adj)
    return np.array(pad_feature_list),np.array(pad_adj_list)


def padding_zeros1(feature_list,len_list):
    max_len=max(len_list)
    pad_features=np.zeros([max_len,feature_list[0].shape[1]])
    pad_feature_list=[]


    for i in range(len(len_list)):
        print(i)
        pad_features[:len_list[i],:]=feature_list[i]
        pad_feature_list.append(pad_features)

    return np.array(pad_feature_list)

features_mat,adj_mat=padding_zeros(original_features,adj_list,len_list)
input_features=padding_zeros1(original_features,len_list)
np.save('ligand_features.npy',features_mat)
np.save('ligand_adj.npy',adj_mat)
np.save('full_length.npy',np.array(len_list))
np.save('full_activity.npy',np.array(activity))
original_features1=np.load('original_features.npy')
input_features.shape