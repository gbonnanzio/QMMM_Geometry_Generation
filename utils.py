import MDAnalysis as mda
import numpy as np
from collections import Counter

vdw_radii = {'H':1.2,'C':1.7,'N':1.55,'O':1.52,'S':1.8,'F':1.47,'Br':1.85,'Mg':1.7}
bond_dists ={'C-C':1.53,'C=O':1.23,'C-O':1.35,'C=N':1.35,'C-S':1.71,'S--N':2.45,'N-H':1.05,'C-N':1.382}
threshold = 0.1

def get_atom_position(mol_universe,index):
    position = mol_universe.atoms[index].position
    return position

def get_dist(coord1,coord2):
    d=np.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2+(coord1[2]-coord2[2])**2)
    return d

def kabsch_algorithm(P, Q):
    # Ensure the points are numpy arrays
    P = np.array(P)
    Q = np.array(Q)

    # Step 1: Compute the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Step 2: Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Step 3: Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Step 4: Compute the optimal rotation matrix using SVD
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case handling
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 5: Compute the translation vector
    t = centroid_Q - np.dot(R, centroid_P)

    return R, t

def get_ThDP_indexes(receptor):
    # receptor is the MDAnalysis universe 

    # read in receptor and isolate ThDP (TPP) residue
    tpp_residue = receptor.select_atoms("resname TPP")
    
    # get coordinates of Sulfur atom in ThDP
    S_index = list(tpp_residue.types).index('S')
    S_coords = tpp_residue.positions[S_index]

    # Identify the carbanion of ThDP
    # iterate through carbon atoms
    ThDP_C_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'C']
    potential_carbanion_indexes = []
    for i in ThDP_C_indexes:
        curr_dist = get_dist(S_coords,tpp_residue.positions[i])
        if (abs(curr_dist - bond_dists['C-S']) < threshold):
            potential_carbanion_indexes.append(i)

    ThDP_N_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'N']

    # find the nitrogen that is proximal to the carbanion
    for i in ThDP_N_indexes:
        curr_dist = get_dist(S_coords,tpp_residue.positions[i])
        if (abs(curr_dist - bond_dists['S--N']) < threshold):
            N_ring_index = i
    N_ring_coords = tpp_residue.positions[N_ring_index]

    # find the primary amine 
    ThDP_H_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'H']
    potential_primary_N_indexes = []
    for i in ThDP_N_indexes:
        for j in ThDP_H_indexes:
            curr_dist = get_dist(tpp_residue.positions[i],tpp_residue.positions[j])
            if (abs(curr_dist - bond_dists['N-H']) < threshold):
                potential_primary_N_indexes.append(i)

    primary_N_trimmed = list(set(potential_primary_N_indexes))
    if len(primary_N_trimmed) > 1:
        print("UH OH! Multiple amino groups found")
    else:
        primary_N_index = primary_N_trimmed[0]

    # find the carbanion
    for i in potential_carbanion_indexes:
        S_dist = get_dist(S_coords,tpp_residue.positions[i])
        S_err = abs(S_dist - bond_dists['C-S'])
        N_dist = get_dist(N_ring_coords,tpp_residue.positions[i])
        N_err = abs(N_dist - bond_dists['C=N'])

        if ((S_err < threshold) and (N_err < threshold)):
            Carbanion_index = i
    
    ThDP_important_indexes = {'C1':Carbanion_index,'N1':N_ring_index,'S1':S_index,'N2':primary_N_index}
    return ThDP_important_indexes

def get_substrate_aka_indexes(substrate):
    # substrate is the substrate mdanalysis universe 

    # Identify the atoms of the alpha-keto acid head for the substrates
    # {'Carbonyl_Carbon':index...}

    # get all C's in list 
    substrate_atoms = substrate.select_atoms('all')
    substrate_atom_types = list(substrate_atoms.types)
    substrate_C_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'C']
    substrate_O_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'O']

    # carbonyl group
    potential_carbonyl_pairs = {} # {carbon index:[oxygen indexes]}
    potential_carboxylic_acid_pairs = {}
    for i in substrate_C_indexes:
        curr_C_coords = substrate_atoms.positions[i]
        for j in substrate_O_indexes:
            curr_O_coords = substrate_atoms.positions[j]
            curr_dist = get_dist(curr_C_coords,curr_O_coords)
            C_d_O_err = abs(curr_dist - bond_dists['C=O']) # double bond

            if C_d_O_err < threshold: # we have found a C=O bond
                true_carbonyl = True
                for k in substrate_O_indexes: # check if this is a carboxylic acid
                    if k != j:
                        curr_O_coords = substrate_atoms.positions[k]
                        curr_dist = get_dist(curr_C_coords,curr_O_coords)
                        C_s_O_err = abs(curr_dist - bond_dists['C-O']) # single bond

                        if C_s_O_err < threshold: # this is a carboxylic acid not a carbonyl
                            potential_carboxylic_acid_pairs[i] = [j,k]
                            true_carbonyl = False

                if true_carbonyl:
                    if i not in potential_carbonyl_pairs:
                        potential_carbonyl_pairs[i] = [j]
                    else:
                        potential_carbonyl_pairs[i].append(j)
    
    potential_carbonyl_carbons = []
    for curr_C_index in potential_carbonyl_pairs:
        potential_oxygens = potential_carbonyl_pairs[curr_C_index]
        if len(potential_oxygens ) == 1:
            potential_carbonyl_carbons.append(curr_C_index)

    if len(potential_carbonyl_carbons) > 1:
        print('UH OH! More than one carbonyl carbon found')
    else:
        carbonyl_carbon_index = potential_carbonyl_carbons[0]
        carbonyl_oxygen_index = potential_carbonyl_pairs[carbonyl_carbon_index][0]

    for i in list(potential_carboxylic_acid_pairs.keys()):
        curr_dist = get_dist(substrate_atoms.positions[carbonyl_carbon_index],substrate_atoms.positions[i])
        C_C_err = abs(curr_dist - bond_dists['C-C'])
        if C_C_err < threshold: 
            carboxylic_acid_carbon_index = i      

    substrate_important_indexes = {'C2':carbonyl_carbon_index,
                                   'O1':carbonyl_oxygen_index,
                                   'C3':carboxylic_acid_carbon_index,
                                   'O2':potential_carboxylic_acid_pairs[carboxylic_acid_carbon_index][0],
                                   'O3':potential_carboxylic_acid_pairs[carboxylic_acid_carbon_index][1]}


    for i in range(0,len(substrate_atoms)):
        if i not in substrate_important_indexes.values():
            curr_atom_type = substrate_atoms.types[i]
            curr_atom_coords = substrate_atoms.positions[i]

            if curr_atom_type == 'N':
                set_dist = bond_dists['C-N']
            elif curr_atom_type == 'C':
                set_dist = bond_dists['C-C']
            else:
                continue

            curr_dist = get_dist(curr_atom_coords,substrate_atoms.positions[carbonyl_carbon_index])
            dist_err = abs(curr_dist - set_dist) 
            if dist_err < threshold: 
                first_tail_atom_index = i 
    
    substrate_important_indexes['R'] = first_tail_atom_index
    
    return substrate_important_indexes