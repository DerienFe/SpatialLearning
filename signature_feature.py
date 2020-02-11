import numpy as np
import esig.tosig as ts
import copy
from load_mol2_data import *
from load_6angstroms_data import *
def get_paths(node_index, adjacent_matrix, L):
    """
    function: compute all the integer encoded paths in the substructure around the atom with in radius L
    Input:
    node_index: the index of the atom
    adjacent_matrix: matrix shows the connection between atoms
    L: lenth of the path
    Output: a list which consists of all the integer encoded paths
    """
    paths = []
    path = np.zeros(L + 1, int)
    path[0] = node_index
    if L == 0:
        paths.append(path)
    else:
        if L == 1:
            temp = np.nonzero(adjacent_matrix[node_index, :])
            for index in list(temp[0]):
                path[1] = index
                paths.append(copy.deepcopy(path))
        else:
            paths_old = get_paths(node_index, adjacent_matrix, L - 1)
            for sub_path in paths_old:
                temp = list(np.nonzero(adjacent_matrix[sub_path[-1], :])[0])
                temp.remove(sub_path[-2])  # delete the self-visited path - sub_path[-2] can't be new segment
                for e in temp:
                    path[:-1] = sub_path
                    path[-1] = e
                    paths.append(copy.deepcopy(path))
    return paths




def categorical_path(label_path,mol,cat_dim,index_map,L):
    """function transform the integer encoded path to categorical path
    input: label_path: the integer encoded path
           mol:molecule information dictionary
           cat_dim: number of categories
           index_map: the map assign the label to the index in categorical path matrix
           L: length of path
    output: a matrix representation of categorical path, of shape cat_dim*(L+2)
    """
    # each categorical path is stored in (5,L+2) matrix
    mat = np.zeros([cat_dim, L + 2])
    # assign the label to each index

    for i in range(L + 1):
        atom_type = mol['atoms'][label_path[i]].ntype
        if i == 0:
            mat[index_map[atom_type], i + 1] = 1
        else:
            temp = np.zeros([cat_dim])
            temp[index_map[atom_type]] = 1
            mat[:, i + 1] = mat[:, i] + temp
    return mat




def categorical_path_sig(node_index, mol,adjacent_matrix, index_map, L, cat_dim, sig_deg,flag):
    """function will output the signatures/logsignatures of categorical path of a node up to radius L
    input: node_index
           adjacent matrix
        mol:molecule information dictionary
           index_map: a dictionary assign atom labels to each index of categorical path
           L: radius around the node
           cat_dim: number of atom types
           sig_deg:degree of signature
           flag: 0 return signature, 1 return log-signature
    output: signatures/logsignatures of the all categorical paths of the node up to length L
            has shape (m,n), m depends on the number of paths at each L, n depends on the cat_dim and sig_deg


    """
    sig_dim = ts.sigdim(cat_dim, sig_deg)
    logsig_dim = ts.logsigdim(cat_dim, sig_deg)
    signatures = None
    log_signatures = None
    if (flag==0):
       for i in range(L):
           paths = get_paths(node_index, adjacent_matrix, i + 1)
           num_path = len(paths)
           paths_sig = np.zeros([num_path, sig_dim],dtype=float)
           k = 0

           for path in paths:
               mat=categorical_path(path,mol,cat_dim,index_map,i)
               paths_sig[k,] = ts.stream2sig(np.transpose(mat), sig_deg)
               k = k + 1
           if signatures is None:
                signatures = paths_sig
           else:
                signatures = np.concatenate(([signatures, paths_sig]), axis=0)
       return(signatures)
    elif(flag==1):
        for i in range(L):
            paths = get_paths(node_index, adjacent_matrix, i + 1)
            num_path = len(paths)
            paths_logsig = np.zeros([num_path, logsig_dim],dtype=float)
            k = 0

            for path in paths:
                mat = categorical_path(path, mol,cat_dim, index_map, i)
                paths_logsig[k,] = ts.stream2logsig(np.transpose(mat), sig_deg)
                k = k + 1
            if log_signatures is None:
               log_signatures = paths_logsig
            else:
               log_signatures = np.concatenate(([log_signatures, paths_logsig]), axis=0)
        return log_signatures




def coordinate_path_sig(node_index,mol,adjacent_matrix,L,sig_deg,flag):
    """
    function will output the signatures/logsignatures of coordinate path of a node up to radius L
    Input: node_index: starting node
           adjacent_matrix: the matrix shows the connection between atoms
           mol:molecule information dictionary
           L: length of path
           sig_deg: degree of signature
           flag: 0 return signature, 1 return log_signature
    output: signatures/logsignatures of the all coordinate paths of the node up to length L
            has shape (m,n), m depends on the number of paths at each L, n= sig_dim/logsig_dim
    """
    sig_dim = ts.sigdim(3, sig_deg)
    logsig_dim = ts.logsigdim(3, sig_deg)
    signatures = None
    log_signatures = None
    if (flag==0):
       for i in range(L):
           paths = get_paths(node_index, adjacent_matrix, i + 1)
           num_path = len(paths)
           paths_sig = np.zeros([num_path, sig_dim])
           #paths_logsig = np.zeros([num_path, logsig_dim])
           k=0
           for path in paths:
               temp=np.zeros([len(path),3])
               for j in range(len(path)):
                   temp[j,:]=mol['atoms'][path[j]].features
               paths_sig[k,] = ts.stream2sig(temp, sig_deg)
               #paths_logsig[k,] = ts.stream2logsig(temp, sig_deg)
               k=k+1
           if signatures is None:
              signatures = paths_sig
           else:
              signatures = np.concatenate(([signatures, paths_sig]), axis=0)
       return(signatures)
    if (flag==1):
        for i in range(L):
            paths = get_paths(node_index, adjacent_matrix, i + 1)
            num_path = len(paths)
            paths_logsig = np.zeros([num_path, logsig_dim])
            k = 0
            for path in paths:
                temp = np.zeros([len(path), 3])
                for j in range(len(path)):
                    temp[j, :] = mol['atoms'][path[j]].features
                paths_logsig[k,] = ts.stream2logsig(temp, sig_deg)
                k = k + 1
            if log_signatures is None:
               log_signatures = paths_logsig
            else:
               log_signatures = np.concatenate(([log_signatures, paths_logsig]), axis=0)
        return log_signatures

"""compute signature feature a molecular level"""
def mol_features(file_dir, index_map, L, cat_dim, deg_sig):
    """

    :param file_dir: the location of the mol2 file
    :param index_map: predefinded index map between different atom types and indices
    :param L: length of the path
    :param cat_dim: number of categories corresponding to number of different atom types
    :param deg_sig: degree of signatures
    :return:
    """
    mol = load_atom(file_dir)
    adjacent_matrix = load_adjacent_matrix(file_dir, mol)
    expected_cat_sig = []
    expected_cat_logsig = []
    expected_xyz_sig = []
    expected_xyz_logsig = []
    for node_index in range(mol['mol_info'].num_atoms):
        cat_sig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=0)
        cat_logsig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=1)
        xyz_sig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=0)
        xyz_logsig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=1)
        expected_cat_sig.append(np.average(cat_sig, 0))
        expected_cat_logsig.append(np.average(cat_logsig, 0))
        expected_xyz_sig.append(np.average(xyz_sig, 0))
        expected_xyz_logsig.append(np.average(xyz_logsig, 0))
    mol_features = {'expected_cat_sig': np.array(expected_cat_sig), 'expected_cat_logsig': np.array(expected_cat_logsig),
                    'expected_xyz_sig': np.array(expected_xyz_sig), 'expected_xyz_logsig': np.array(expected_xyz_logsig)}
    return mol_features

def mol_KdKi_features(file_dir, index_map, L, cat_dim, deg_sig,flag):
    """

    :param file_dir: the location of the mol2 file
    :param index_map: predefinded index map between different atom types and indices
    :param L: length of the path
    :param cat_dim: number of categories corresponding to number of different atom types
    :param deg_sig: degree of signatures
    :return:
    """
    mol = load_KdKi_data(file_dir)
    adjacent_matrix = load_KdKi_adj(file_dir, mol)
    expected_cat_sig = []
    expected_cat_logsig = []
    expected_xyz_sig = []
    expected_xyz_logsig = []
    for node_index in range(mol['mol_info'].num_atoms):
        if flag=='sig':
          cat_sig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=0)
          xyz_sig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=0)
          expected_cat_sig.append(np.average(cat_sig, 0))
          expected_xyz_sig.append(np.average(xyz_sig, 0))
          mol_features = {'expected_cat_sig': np.array(expected_cat_sig),
                          'expected_xyz_sig': np.array(expected_xyz_sig),}
        elif flag=='logsig':
           cat_logsig = categorical_path_sig(node_index, mol, adjacent_matrix, index_map, L, cat_dim, deg_sig, flag=1)
           xyz_logsig = coordinate_path_sig(node_index, mol, adjacent_matrix, L, deg_sig, flag=1)
           expected_cat_logsig.append(np.average(cat_logsig, 0))
           expected_xyz_logsig.append(np.average(xyz_logsig, 0))
           mol_features = {'expected_cat_logsig': np.array(expected_cat_logsig),
                           'expected_xyz_logsig': np.array(expected_xyz_logsig)}
    return mol_features
if __name__ == '__main__':
    """example to compute signature features from PDB file with assigned adjacent matrix"""
    """
    from load_data import *
    #import data from database
    
    db_dir = "database/"  # database directory
    ligand, protein, water = load_data1(db_dir)
    coordinate_data = get_3D_coordinates(ligand)
    xdata = coordinate_data[:, 0]
    ydata = coordinate_data[:, 1]
    zdata = coordinate_data[:, 2]
    color_map = {'C': 'r',
                 'O': 'g',
                 'N': 'y',
                 'S': 'b',
                 'P': 'm'}
    #design a adjacent matrix
    L = len(ligand)
    adjacent_matrix = np.zeros((L, L), int)
    for i in range(L):
        for j in range(L):
            if (np.abs(j - i) < 3 and (i != j)):
                adjacent_matrix[i, j] = 1
    print("adjacent matrix=",adjacent_matrix)

    # visualize the paths of an atom for given L
    node_index = 0
    L = 2
    paths = get_paths(node_index, adjacent_matrix, L)
    print("paths from node 0 with length 2",paths)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    path = paths[-1]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # for path in paths:
    for i in range(len(path) - 1):
        ax.plot(xdata[[path[i], path[i + 1]]], ydata[[path[i], path[i + 1]]], zdata[[path[i], path[i + 1]]], color='b')

    # for path in paths:
    for i in path:
        atom_t = ligand[i].ntype
        ax.scatter3D(xdata[i], ydata[i], zdata[i], s=50, color=color_map[atom_t])
        ax.text(xdata[i] + 0.2, ydata[i], zdata[i], atom_t, style='italic')
    ax.grid(False)
    plt.show()

    # visualize an example of a categorical path
    path=paths[-1]
    L=2
    # assign the label to each index
    index_map = {'C': 0,
                 'O': 1,
                 'N': 2,
                 'S': 3,
                 'P': 4}

    cate_path=categorical_path(path,5,index_map,L)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # path_img
    for i in range(L + 1):
        ax.plot([cate_path[0, i], cate_path[0, i + 1]], [cate_path[2, i], cate_path[2, i + 1]], [cate_path[1, i], cate_path[1, i + 1]], 'go--', color='b')
        #print([cate_path[0, i], cate_path[0, i + 1]], [cate_path[2, i], cate_path[2, i + 1]], [cate_path[1, i], cate_path[1, i + 1]])
    ax.set_xlabel('C', fontsize=10, color='b')
    ax.set_ylabel('N', fontsize='medium', color='r')
    ax.set_zlabel('O', fontsize='medium', color='g')
    plt.show()

    #cateorical path
    cat_sig=categorical_path_sig(node_index,adjacent_matrix,index_map,3,5,3,0)
    cat_logsig=categorical_path_sig(node_index,adjacent_matrix,index_map,3,5,3,1)
    print("categorical_signature=",cat_sig)
    print("categorical_logsignature=",cat_logsig)

    #coordinate path
    xyz_sig = coordinate_path(node_index, adjacent_matrix, 3, 3,0)
    xyz_logsig = coordinate_path(node_index, adjacent_matrix, 3, 3,1)
    print("coordinate_signature=",xyz_sig)
    print("coordingate_logsignature=",xyz_logsig)




    #example to compute signature features from mol2 file
    from load_mol2_data import *
    dir = "mol2 structures/"
    import os

    dirlist = os.listdir(dir)
    file_dir=dir + dirlist[0]
    mol=load_atom(file_dir)
    adjacent_matrix=load_adjacent_matrix(file_dir)
    coordinate_data = get_3D_coordinates(mol)
    xdata = coordinate_data[:, 0]
    ydata = coordinate_data[:, 1]
    zdata = coordinate_data[:, 2]
    # visualize the paths of an atom for given L
    node_index = 10
    L = 3
    paths = get_paths(node_index, adjacent_matrix, L)
    print("paths from node 0 with length 2", paths)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    color_map = {'C': 'r',
                 'O': 'g',
                 'H': 'y',
                 'N': 'b',
                 'S': 'm'}

    path = paths[1]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # for path in paths:
    for i in range(len(path) - 1):
        ax.plot(xdata[[path[i], path[i + 1]]], ydata[[path[i], path[i + 1]]], zdata[[path[i], path[i + 1]]], color='b')
    # for path in paths:
    for i in path:
        atom_t = mol[i].ntype
        ax.scatter3D(xdata[i], ydata[i], zdata[i], s=50, color=color_map[atom_t])
        ax.text(xdata[i] + 0.2, ydata[i], zdata[i], atom_t, style='italic')
    ax.grid(False)
    plt.show()

    # visualize an example of a categorical path
    path = paths[1]
    L = 3
    # assign the label to each index
    index_map = {'C': 0,
                 'O': 1,
                 'H': 2,
                 'N': 3,
                 'S': 4}

    cate_path = categorical_path(path,mol, 5, index_map, L)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # path_img
    for i in range(L + 1):
        ax.plot([cate_path[0, i], cate_path[0, i + 1]], [cate_path[2, i], cate_path[2, i + 1]],
                [cate_path[1, i], cate_path[1, i + 1]], 'go--', color='b')
        # print([cate_path[0, i], cate_path[0, i + 1]], [cate_path[2, i], cate_path[2, i + 1]], [cate_path[1, i], cate_path[1, i + 1]])
    ax.set_xlabel('C', fontsize=10, color='b')
    ax.set_ylabel('H', fontsize='medium', color='r')
    ax.set_zlabel('O', fontsize='medium', color='g')
    plt.show()

    # cateorical path
    cat_sig = categorical_path_sig(node_index,mol,adjacent_matrix, index_map, 3, 5, 3, 0)
    cat_logsig = categorical_path_sig(node_index,mol, adjacent_matrix, index_map, 3, 5, 3, 1)
    #print("categorical_signature=", cat_sig)
    #print("categorical_logsignature=", cat_logsig)
    print(cat_sig.shape)
    
    expected_cat_sig=np.average(cat_sig,0)
    expected_cat_logsig = np.average(cat_logsig, 0)
    print("expected categorical_siganture=",expected_cat_sig)
    print("expected categorical_logsiganture=", expected_cat_logsig)

    # coordinate path
    xyz_sig = coordinate_path_sig(node_index,mol, adjacent_matrix, 3, 3, 0)
    xyz_logsig = coordinate_path_sig(node_index, mol,adjacent_matrix, 3, 3, 1)
    #print("coordinate_signature=", xyz_sig)
    #print("coordingate_logsignature=", xyz_logsig)
    print(xyz_sig.shape)
    expected_xyz_sig = np.average(xyz_sig, 0)
    expected_xyz_logsig = np.average(xyz_logsig, 0)
    print("expected coordinate_siganture=",expected_xyz_sig)
    print("expected coordinate_siganture=", expected_xyz_logsig)
    """