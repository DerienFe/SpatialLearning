import numpy as np
from molecule_lib import Node,Mol


"""read mol2 file and parse data"""


def load_atom(db_dir):
    """load the atom information includes atom type, coordinates and
    atom index form mol2 file"""
    current = open(db_dir, "r")
    mol2_file = []
    for row in current:
        line = row.split()
        mol2_file.append(line)
    atom_start = mol2_file.index(['@<TRIPOS>ATOM']) + 1
    atom_end = mol2_file.index(['@<TRIPOS>BOND'])
    atom_info=mol2_file[atom_start:atom_end]
    atoms=[]
    atom_list=[]
    for line in atom_info:
        atom_type = line[1][0]
        x_y_z = np.asarray(line[2:5], float)
        idx = int(line[0])
        node1 = Node(atom_type, x_y_z, idx)
        atoms.append(node1)
        if atom_type not in atom_list:
            atom_list.append(atom_type)
    mol_info=Mol(idx,atom_list)
    mol={'mol_info':mol_info,'atoms':atoms}
    return mol

def load_adjacent_matrix(db_dir,mol):
    """load the atom information includes atom type, coordinates and
    atom index form mol2 file"""
    current = open(db_dir, "r")
    mol2_file = []
    for row in current:
        line = row.split()
        mol2_file.append(line)
    bond_start = mol2_file.index(['@<TRIPOS>BOND']) + 1
    bond_end = mol2_file.index(['@<TRIPOS>SUBSTRUCTURE'])
    bond_info=mol2_file[bond_start:bond_end]
    adjacent_matrix=np.zeros([mol['mol_info'].num_atoms,mol['mol_info'].num_atoms])
    for line in bond_info:
        adjacent_matrix[int(line[1])-1,int(line[2])-1] = 1
        adjacent_matrix[int(line[2])-1, int(line[1])-1] = 1
    return adjacent_matrix


def get_3D_coordinates(ligand):
    coordinate_data = np.empty((0, 3), float)
    for i in ligand:
        x_y_z = i.features.reshape([1, 3])
        coordinate_data = np.concatenate([coordinate_data, x_y_z], 0)
    return coordinate_data