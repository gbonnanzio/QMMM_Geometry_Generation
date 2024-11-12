import MDAnalysis as mda
import numpy as np
from collections import Counter
from scipy.optimize import minimize

vdw_radii = {'H':1.2,'C':1.7,'N':1.55,'O':1.52,'S':1.8,'F':1.47,'Br':1.85,'Mg':1.7}
bond_dists ={'C-C':1.53,'C=O':1.23,'C-O':1.35,'C=N':1.35,'C-S':1.71,'S--N':2.45,'N-H':1.05,'C-N':1.382}
threshold = 0.1

def get_atom_position(mol_universe,index):
    position = mol_universe.atoms[index].position
    return position

def get_dist(coord1,coord2):
    d=np.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2+(coord1[2]-coord2[2])**2)
    return d

def get_angle(coord1,coord2,coord3):
        # Calculate vectors BA and BC
        BA = coord1 - coord2
        BC = coord3 - coord2

        # Compute the dot product and magnitudes of BA and BC
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_BA * magnitude_BC)

        # Handle potential floating-point errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians and then convert to degrees
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

def atom_objective(point, centers, radii):
    return sum((np.linalg.norm(point - center) - radius)**2 for center, radius in zip(centers, radii))

def combined_objective(all_points, centers, radii):
    # Split all_points into the three key points
    C2, C3, O1 = np.split(all_points, len(all_points)/3)
    
    # Calculate atom-level errors for each point to fit them to sphere constraints
    atom_err_C2 = atom_objective(C2, centers, radii[0])
    atom_err_C3 = atom_objective(C3, centers, radii[1])
    atom_err_O1 = atom_objective(O1, centers, radii[2])
    
    total_atom_err = atom_err_C2 + atom_err_C3 + atom_err_O1
    
    # Combine errors with weighting factors
    return  total_atom_err 

def optimize_points(centers, initial_guess, radii):
    # Flatten initial guess as midpoint of each set of centers
    # Set up optimization
    tolerance = 1e-6
    result = minimize(
        combined_objective,
        initial_guess,
        args=(centers, radii),
        tol=tolerance
    )
    C2, C3, O1 = np.split(result.x, 3)
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    return C2, C3, O1


def angle_objective(atom1,atom2,atom3,angle):
    angle_err = abs(get_angle(atom1,atom2,atom3) - angle)
    return angle_err

def combined_angle_objective(tail_coord, C1_coord, C2_coord, O1_coord, C3_coord, angles):
    # Calculate atom-level errors for each point to fit them to sphere constraints
    angle_err_C1_C2_R = angle_objective(C1_coord, C2_coord, tail_coord, angles[0])
    angle_err_O1_C2_R = angle_objective(O1_coord, C2_coord, tail_coord, angles[1])
    angle_err_C3_C2_R = angle_objective(C3_coord, C2_coord, tail_coord, angles[2])
    
    #dist_err = get_dist()

    total_angle_err = angle_err_C1_C2_R + angle_err_O1_C2_R + angle_err_C3_C2_R
    # Combine errors with weighting factors
    return  total_angle_err 

def optimize_angles(initial_guess,C1_coords,C2_coords,O1_coords,C3_coords,angles):
    # Flatten initial guess as midpoint of each set of centers
    # Set up optimization
    tolerance = 1e-6
    max_dist = 1.382
    bounds = [(C2_coords[0]-max_dist,C2_coords[0]+max_dist),(C2_coords[1]-max_dist,C2_coords[1]+max_dist),(C2_coords[2]-max_dist,C2_coords[2]+max_dist)]
    result = minimize(
        combined_angle_objective,
        initial_guess,
        args=(C1_coords,C2_coords,O1_coords,C3_coords,angles),
        tol=tolerance,
        bounds=bounds
    )
    
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    
    return result.x

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

def get_ThDP_indexes(tpp_residue):
    
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

import MDAnalysis as mda
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_atoms(universe, atom1_idx, atom2_idx, atom_list, angle_deg):
    """
    Rotates atoms in `atom_list` around the axis defined by `atom1_idx` and `atom2_idx` by `angle_deg`.
    
    Parameters:
    - universe: MDAnalysis Universe object
    - atom1_idx: Index of the first fixed atom
    - atom2_idx: Index of the second fixed atom
    - atom_list: List of atom indices to be rotated
    - angle_deg: Angle of rotation in degrees
    """
    # Get coordinates of fixed atoms
    atom1 = universe.atoms[atom1_idx]
    atom2 = universe.atoms[atom2_idx]
    coord1 = atom1.position
    coord2 = atom2.position
    
    # Define the rotation axis (normalized vector between atom1 and atom2)
    axis = coord2 - coord1
    axis = axis / np.linalg.norm(axis)

    # Define rotation angle in radians
    angle_rad = np.deg2rad(angle_deg)

    # Create the rotation object
    rotation = R.from_rotvec(angle_rad * axis)

    # Rotate each atom in the atom_list
    for atom_idx in atom_list:
        atom = universe.atoms[atom_idx]
        # Translate the atom to the origin (relative to atom1)
        atom_position = atom.position - coord1
        # Apply rotation
        rotated_position = rotation.apply(atom_position)
        # Translate back to original frame
        atom.position = rotated_position + coord1


