import MDAnalysis as mda
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import os
import csv

bond_dists ={'C-C':1.53,'C=C':1.35,'long C-C':1.64,'CO2 C-O':1.27,'C=O':1.23,'C-O':1.35,'C=N':1.35,'C-S':1.71,'S--N':2.45,'N-H':1.05,'C-N':1.382,'O-H':1.05}
threshold = 0.1

def get_atom_clashes(universe,atom_indices,threshold):
    # Create an AtomGroup for the specified atom indices
    nearby_atoms = universe.select_atoms(f"protein and (around " + str(threshold) + f" index {' '.join(map(str, atom_indices))})")
    return len(nearby_atoms)

def get_atom_position(mol_universe:mda.core.universe.Universe,index:int):
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
    C2_coords, C3_coords, O1_coords = np.split(all_points, len(all_points)/len(radii))
    
    # Calculate atom-level errors for each point to fit them to sphere constraints
    atom_err_C2 = atom_objective(C2_coords, centers, radii[0])
    atom_err_C3 = atom_objective(C3_coords, centers, radii[1])
    atom_err_O1 = atom_objective(O1_coords, centers, radii[2])
    
    total_atom_err = atom_err_C2 + atom_err_C3 + atom_err_O1
    
    return  total_atom_err 

def optimize_coordinates(initial_guess:np.ndarray, centers:np.ndarray, radii:list):
    '''
    OBJECTIVE:
        Find the coordinates of three points in space given the centers of four spheres
        centered around reference atoms: C1, N1, N2, S1. 
        The radii were picked such that the intersection of these spheres will be the 
        coordinates of each atom C2, C3, O1
        initial_guess = [C2_x, C2_y, C2_z, C3_x, C3_y...] (needs to be 1D for optimizer)
        centers = [[C1_x,C1_y,C1_z], 
                   [N1_x,N1_y,N1_z],
                   [N2_x,N2_y,N2_z],
                   [S1_x,S1_y,S1_z]]
        radii = [[r_C2_C1, r_C2_N1, r_C2_N2, r_C2_S1],
                 [r_C3_C1, r_C3_N1, r_C3_N2, r_C3_S1]]
                 [r_O1_C1, r_O1_N1, r_O1_N2, r_O1_S1]]
    '''
    # Set up optimization
    tolerance = 1e-6
    result = minimize(
        combined_objective,
        initial_guess,
        args=(centers, radii),
        tol=tolerance
    )
    C2_coords, C3_coords, O1_coords = np.split(result.x, 3)
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    return C2_coords, C3_coords, O1_coords

def angle_objective(atom1,atom2,atom3,angle):
    angle_err = abs(get_angle(atom1,atom2,atom3) - angle)
    return angle_err

def combined_angle_objective(tail_coord, C1_coord, C2_coord, O1_coord, C3_coord, angles):
    # Calculate atom-level errors for each point to fit them to sphere constraints
    angle_err_C1_C2_R = angle_objective(C1_coord, C2_coord, tail_coord, angles[0])
    angle_err_O1_C2_R = angle_objective(O1_coord, C2_coord, tail_coord, angles[1])
    angle_err_C3_C2_R = angle_objective(C3_coord, C2_coord, tail_coord, angles[2])
    
    total_angle_err = angle_err_C1_C2_R + angle_err_O1_C2_R + angle_err_C3_C2_R
    return  total_angle_err 

def optimize_angles(initial_guess,C1_coords,C2_coords,O1_coords,C3_coords,angles):
    '''
    OBJECTIVE: 
    Identify the coordinate of atom R (initial_guess) that lead to the angles
    C1-C2-R, O1-C2-R, C3-C2-R given in angles
    '''
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

def write_universe(directory:str, filename:str, universe:mda.core.universe.Universe):
    '''
    OBJECTIVE: 
    Write an MDanalysis atom universe to directory+filename. If none exist,
    create that directory  
    '''
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating it...")
        os.makedirs(directory)
    
    # Write the file
    file_path = os.path.join(directory, filename)
    universe.atoms.write(file_path)
    print(f"File '{filename}' has been written in '{directory}'.")

def edit_protein_files(directory,file_name):
    '''
    OBJECTIVE:
    Edit the protein file to search for protein subunits (ending in OXT)
    to add "TER" needed for this pdb file to be amber readable
    '''
    file_path = directory + file_name
    # edit the protein file to add the TER labels needed for Amber forcefield generation 
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
    with open(file_path, 'w') as outfile:
        for line in lines:
            outfile.write(line)
            if " OXT " in line:  # Check if the line contains the atom name "OXT"
                outfile.write("TER\n")
    print('Edited ',file_path,' for Amber')

def determine_index_shift(original_u,merged_u,original_indexes):
    num_atom_diff = len(merged_u.atoms)-len(original_u.atoms)
    # if original atoms were added first
    if np.all(merged_u.atoms[0].position == original_u.atoms[0].position):
        return 0
    elif np.all(merged_u.atoms[num_atom_diff].position == original_u.atoms[0].position):
        return num_atom_diff
    else:
        return None

def kabsch_algorithm(P:np.ndarray, Q:np.ndarray):
    '''
    OBJECTIVE:
        Given two sets of multiple coordinates (at least three), find the 
        rotational and translational matrix you would apply to P to get to Q
    '''
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

def get_ThDP_indexes(tpp_residue:mda.core.universe.Universe):
    '''
    OBJECTIVE:
    Get the important atom indexes of ThDP
        C1 = Carbanion
        S1 = Sulfur
        N1 = Ring Nitrogen
        N2 = Amino group 
    '''
    # Get coordinates of S1 (only sulfur in ThDP)
    S1_index = list(tpp_residue.types).index('S')
    S1_coords = tpp_residue.positions[S1_index]

    # Identify the carbanion of ThDP by distance away from S and N


    # find N1 by distance away from S
    ThDP_N_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'N']
    min_error = 1000
    found_N1 = False
    for i in ThDP_N_indexes:
        curr_dist = get_dist(S1_coords,tpp_residue.positions[i])
        curr_error = abs(curr_dist - bond_dists['S--N'])
        if curr_error < min_error:
            min_error = curr_error
            min_index = i
        if curr_error < threshold:
            found_N1 = True
            N1_index = i
    
    if found_N1 == False:
        print('Taking best guess at ring N')
        N1_index = min_index
    
    N1_coords = tpp_residue.positions[N1_index]

    # find N2, should be only N with two hydrogens on it  
    ThDP_H_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'H']
    potential_N2_indexes = []
    for i in ThDP_N_indexes:
        for j in ThDP_H_indexes:
            curr_dist = get_dist(tpp_residue.positions[i],tpp_residue.positions[j])
            if (abs(curr_dist - bond_dists['N-H']) < threshold):
                potential_N2_indexes.append(i)
    # Count occurrences
    counter = Counter(potential_N2_indexes)
    # Filter the list to keep only elements that occur exactly twice
    N2_trimmed = list(set([item for item in potential_N2_indexes if counter[item] == 2]))
    #N2_trimmed = list(set(potential_N2_indexes))
    if len(N2_trimmed) > 1:
        print("UH OH! Multiple amino groups found")
    else:
        N2_index = N2_trimmed[0]

    # We have now found, S1, N1, and N2... use S1 and N1 to find C1
    min_error = 1000
    found_C1 = False
    # iterate through carbon atoms
    ThDP_C_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'C']
    for i in ThDP_C_indexes:
        S_dist = get_dist(S1_coords,tpp_residue.positions[i])
        S_err = abs(S_dist - bond_dists['C-S'])
        N_dist = get_dist(N1_coords,tpp_residue.positions[i])
        N_err = abs(N_dist - bond_dists['C=N'])
        
        curr_error = S_err + N_err
        if curr_error < min_error:
            min_error = curr_error
            min_index = i

        if ((S_err < threshold) and (N_err < threshold)):
            found_C1 = True
            C1_index = i

    if found_C1 == False:
        print('Taking best guess at carbanion')
        C1_index = min_index
    
    ThDP_important_indexes = {'C1':C1_index,'N1':N1_index,'S1':S1_index,'N2':N2_index}
    return ThDP_important_indexes

def get_substrate_aka_indexes(substrate_atoms:mda.core.universe.Universe):
    '''
    OBJECTIVE: 
        Get the important indexes (alpha-keto acid atoms) of the substrate molecule 
        C2 = Carbonyl carbon
        C3 = Carboxylic acid carbon
        O1 = Carbonyl oxygen
        O2/O3 = Oxygens of carboxylic acid 
    '''
    # get all C's in list 
    substrate_atom_types = list(substrate_atoms.types)
    substrate_C_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'C']
    substrate_O_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'O']

    # identify carbon oxygen pairs 
    potential_C2_O1_pairs = {} # {carbon index:[oxygen indexes]}
    potential_C3_O2_O3_pairs = {} # {carbon index:[oxygen indexes]}
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
                            potential_C3_O2_O3_pairs[i] = [j,k]
                            true_carbonyl = False
                # if we found a carbonyl carbon
                if true_carbonyl:
                    potential_C2_O1_pairs[i] = [j]
    
    potential_C2_carbons = []
    for curr_C_index in potential_C2_O1_pairs:
        potential_oxygens = potential_C2_O1_pairs[curr_C_index]
        if len(potential_oxygens ) == 1:
            potential_C2_carbons.append(curr_C_index)

    if len(potential_C2_carbons) > 1:
        print('UH OH! More than one carbonyl carbon was found')
        return None
    else:
        C2_index = potential_C2_carbons[0]
        O1_index = potential_C2_O1_pairs[C2_index][0]

    C2_coords = substrate_atoms.positions[C2_index]
    found_C3 = False
    # Find the carboxylic acid carbons 
    for i in list(potential_C3_O2_O3_pairs.keys()):
        curr_dist = get_dist(C2_coords,substrate_atoms.positions[i]) # C2 and C3 should be 1.53 A apart
        C_C_err = abs(curr_dist - bond_dists['C-C'])
        if C_C_err < threshold: 
            if found_C3 == True:
                print('UH OH! More than one carboxylic acid carbon was found')
                return None
            else:
                C3_index = i    
                O2_index = potential_C3_O2_O3_pairs[C3_index][0]  
                O3_index = potential_C3_O2_O3_pairs[C3_index][1]  
                found_C3 = True

    substrate_important_indexes = {'C2':C2_index,
                                   'O1':O1_index,
                                   'C3':C3_index,
                                   'O2':O2_index,
                                   'O3':O3_index}

    # now find the first atom of the R group of the alpha keto acid 
    for i in range(0,len(substrate_atoms)):
        if i not in substrate_important_indexes.values():
            curr_atom_type = substrate_atoms.types[i]
            curr_atom_coords = substrate_atoms.positions[i]
            # given the original 20 substrates this connecting atom could either be a carbon or a nitrogen 
            if curr_atom_type == 'N':
                set_dist = bond_dists['C-N']
            elif curr_atom_type == 'C':
                set_dist = bond_dists['C-C']
            else: # if it is neither N or C it is not the connecting atom, skip it 
                continue

            curr_dist = get_dist(curr_atom_coords,substrate_atoms.positions[C2_index])
            dist_err = abs(curr_dist - set_dist) 
            if dist_err < threshold: 
                R_index = i 
    
    substrate_important_indexes['R'] = R_index
    
    return substrate_important_indexes

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

    rotated_universe = universe.copy()

    # Rotate each atom in the atom_list
    for atom_idx in atom_list:
        atom = rotated_universe.atoms[atom_idx]
        # Translate the atom to the origin (relative to atom1)
        atom_position = atom.position - coord1
        # Apply rotation
        rotated_position = rotation.apply(atom_position)
        # Translate back to original frame
        atom.position = rotated_position + coord1

    return rotated_universe

def optimize_tail_position(initial_guess,C1_coords,C2_coords,O1_coords,C3_coords,angles):

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

def simplify_integer_list(int_list):
    if not int_list:
        return ""

    # Sort the list to make sure the integers are in ascending order
    sorted_list = sorted(int_list)
    ranges = []
    start = sorted_list[0]
    end = start

    for number in sorted_list[1:]:
        if number == end + 1:
            end = number
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}:{end}")
            start = number
            end = start

    # Add the last range or number
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}:{end}")

    return ' '.join(ranges)

def get_atoms_by_distance(mol_universe:mda.Universe,dist1:float,dist2:float,atom_id:int):
    
    # get atoms a certain distance away from the carbonyl carbon  
    QM_shell= mol_universe.select_atoms("(around " + str(dist1) + " index " + str(atom_id) + ") or resname MG")
    # get the residues those atoms belong to
    QM_residues = set([atom.residue for atom in QM_shell])
    # do the same for the active atoms (which should include the QM atoms) 
    active_atoms = mol_universe.select_atoms("(around " + str(dist2) + " index " + str(atom_id) + ") or resname MG")
    active_residues = set([atom.residue for atom in active_atoms])
    print(active_residues)

    # get all the atoms that belong to the residues for each of these groups 
    QM_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue.resid}" for residue in QM_residues]))
    QM_atoms_indexes = [curr_atom.index for curr_atom in QM_atoms]
    active_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue.resid}" for residue in active_residues]))
    active_atoms_indexes = [curr_atom.index for curr_atom in active_atoms]
    # find the fixed atoms 
    fixed_atoms = mol_universe.select_atoms(f"around 2.5 index {' '.join(map(str, active_atoms_indexes))}")
    fixed_atoms_indexes = [curr_atom.index for curr_atom in fixed_atoms]
    
    return QM_residues, active_residues, QM_atoms_indexes,active_atoms_indexes,fixed_atoms_indexes

def get_water_by_distance(mol_universe:mda.Universe,dist1:float,dist2:float,dist3:float,atom_id:int):
    
    # get atoms a certain distance away from the carbonyl carbon  
    QM_shell= mol_universe.select_atoms("(around " + str(dist1) + " index " + str(atom_id) + ") or resname MG")
    # get the residues those atoms belong to
    QM_residues = set([atom.residue for atom in QM_shell])
    # do the same for the active atoms (which should include the QM atoms) 
    selection = "(around "+str(dist2)+" index "+str(atom_id)+") or (resname WAT and (around "+str(dist3)+" index "+str(atom_id)+")) or resname MG"
    active_atoms = mol_universe.select_atoms(selection)
    active_residues = set([atom.residue for atom in active_atoms])
    print(active_residues)
    # get all the atoms that belong to the residues for each of these groups 
    QM_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue.resid}" for residue in QM_residues]))
    QM_atoms_indexes = [curr_atom.index for curr_atom in QM_atoms]
    active_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue.resid}" for residue in active_residues]))
    active_atoms_indexes = [curr_atom.index for curr_atom in active_atoms]
    # find the fixed atoms 
    fixed_atoms = mol_universe.select_atoms(f"around 2.5 index {' '.join(map(str, active_atoms_indexes))}")
    fixed_atoms_indexes = [curr_atom.index for curr_atom in fixed_atoms]
    
    return QM_residues, active_residues, QM_atoms_indexes,active_atoms_indexes,fixed_atoms_indexes

def get_atoms_by_reslist(mol_universe:mda.Universe,residue_list:list):

    # get all the atoms that belong to the residues for each of these groups 
    selected_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue}" for residue in residue_list]))
    selected_atoms_indexes = [curr_atom.index for curr_atom in selected_atoms]

    return selected_atoms_indexes

def find_adjacent_oxygens(atom_universe,carbon_index):
    
    carbon_coords = atom_universe.atoms[carbon_index].position

    atom_types = list(atom_universe.types)
    O_indexes = [i for i, x in enumerate(atom_types) if x == 'O']

    adjacent_oxygens = []
    for curr_O_index in O_indexes:
        curr_O_coords = atom_universe.atoms[curr_O_index].position
        curr_dist = get_dist(carbon_coords,curr_O_coords)
        dist_err = abs(curr_dist-bond_dists['CO2 C-O'])
        if dist_err < threshold:
            adjacent_oxygens.append(curr_O_index)

    return adjacent_oxygens

def get_ini_indexes(ini_atoms):
    # ini_atoms is the universe of just the INI residue  

    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(ini_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = ini_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    ini_atom_types = list(ini_atoms.types)
    ini_C_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'C']

    # search for C2, should be C-C distance away from C1 
    potential_C2_indexes = []
    for i in ini_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C-C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = ini_atoms.atoms[C2_index].position

    # search for O1
    ini_O_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'O']
    potential_O1_indexes = []
    for i in ini_O_indexes:
        curr_O_coords = ini_atoms.positions[i]
        curr_dist = get_dist(C2_coords,curr_O_coords)
        curr_err = abs(curr_dist-bond_dists['C-O'])
        if curr_err < threshold:
            potential_O1_indexes.append(i)
    
    if len(potential_O1_indexes) != 1:
        print(potential_O1_indexes)
        print('ERROR FINDING O1 ATOM')
        return None
    else:
        O1_index = potential_O1_indexes[0]
    
    # search for C3
    potential_C3_indexes = []
    for i in ini_C_indexes:
        if i == C1_index or i == C2_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C2_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['long C-C'])
            if curr_err < threshold:
                potential_C3_indexes.append(i)

    if len(potential_C3_indexes) == 0:
        print('ERROR FINDING C3 ATOM, NO ATOMS FOUND')
        return None
    elif len(potential_C3_indexes) == 1:
        C3_index = potential_C3_indexes[0] 
        CO2_oxygens = find_adjacent_oxygens(ini_atoms,C3_index)   
    else:
        refined_C3_indexes = []
        refined_CO2_oxygens = []
        for curr_C3_index in potential_C3_indexes:
            curr_CO2_oxygens = find_adjacent_oxygens(ini_atoms,curr_C3_index) 
            if len(curr_CO2_oxygens) != 2:
                continue
            else:
                refined_C3_indexes.append(curr_C3_index)
                refined_CO2_oxygens.append(curr_CO2_oxygens)

        if len(refined_C3_indexes) != 1:
            print('COULD NOT FIND C3')
            return None
        else:
            C3_index = refined_C3_indexes[0]
            CO2_oxygens = refined_CO2_oxygens[0]
        
    INI_important_indexes = {'C1':C1_index,'C2':C2_index,'C3':C2_index,'O1':O1_index,'CO2_0':CO2_oxygens[0],'CO2_1':CO2_oxygens[1]}
    return INI_important_indexes

def get_ino_protonated_indexes(ino_atoms):
    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(ino_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = ino_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    ino_atom_types = list(ino_atoms.types)
    ino_C_indexes = [i for i, x in enumerate(ino_atom_types) if x == 'C']

    # search for C2, should be C=C distance away from C1 
    potential_C2_indexes = []
    for i in ino_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = ino_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C=C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = ino_atoms.atoms[C2_index].position

    # search for O1
    ini_O_indexes = [i for i, x in enumerate(ino_atom_types) if x == 'O']
    potential_O1_indexes = []
    for i in ini_O_indexes:
        curr_O_coords = ino_atoms.positions[i]
        curr_dist = get_dist(C2_coords,curr_O_coords)
        curr_err = abs(curr_dist-bond_dists['C-O'])
        if curr_err < threshold:
            potential_O1_indexes.append(i)
    
    if len(potential_O1_indexes) != 1:
        print(potential_O1_indexes)
        print('ERROR FINDING O1 ATOM')
        return None
    else:
        O1_index = potential_O1_indexes[0]

    O1_coords = ino_atoms.positions[O1_index]
    # search for H
    ino_H_indexes = [i for i, x in enumerate(ino_atom_types) if x == 'H']
    potential_H_indexes = []
    for i in ino_H_indexes:
        curr_H_coords = ino_atoms.positions[i]
        curr_dist = get_dist(O1_coords,curr_H_coords)
        curr_err = abs(curr_dist-bond_dists['O-H'])
        if curr_err < threshold:
            potential_H_indexes.append(i)
    
    if len(potential_H_indexes) != 1:
        print(potential_H_indexes)
        print('ERROR FINDING H ATOM')
        return None
    else:
        H_index = potential_H_indexes[0]
        
    INO_important_indexes = {'C1':C1_index,'C2':C2_index,'O1':O1_index,'H':H_index}
    return INO_important_indexes   

def get_ino_indexes(ini_atoms):
    # ini_atoms is the universe of just the INI residue  

    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(ini_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = ini_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    ini_atom_types = list(ini_atoms.types)
    ini_C_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'C']

    # search for C2, should be C=C distance away from C1 
    potential_C2_indexes = []
    for i in ini_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C=C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = ini_atoms.atoms[C2_index].position

    # search for O1
    ini_O_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'O']
    potential_O1_indexes = []
    for i in ini_O_indexes:
        curr_O_coords = ini_atoms.positions[i]
        curr_dist = get_dist(C2_coords,curr_O_coords)
        curr_err = abs(curr_dist-bond_dists['C-O'])
        if curr_err < threshold:
            potential_O1_indexes.append(i)
    
    if len(potential_O1_indexes) != 1:
        print(potential_O1_indexes)
        print('ERROR FINDING O1 ATOM')
        return None
    else:
        O1_index = potential_O1_indexes[0]
        
    INO_important_indexes = {'C1':C1_index,'C2':C2_index,'O1':O1_index}
    return INO_important_indexes 

def get_inp_indexes(inp_atoms):
    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(inp_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = inp_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    inp_atom_types = list(inp_atoms.types)
    inp_C_indexes = [i for i, x in enumerate(inp_atom_types) if x == 'C']

    # search for C2, should be C=C distance away from C1 
    potential_C2_indexes = []
    for i in inp_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = inp_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C-C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = inp_atoms.atoms[C2_index].position

    # search for O1
    #inp_O_indexes = [i for i, x in enumerate(inp_atom_types) if x == 'O']
    #potential_O1_indexes = []
    #for i in inp_O_indexes:
    #    curr_O_coords = inp_atoms.positions[i]
    #    curr_dist = get_dist(C2_coords,curr_O_coords)
    #    curr_err = abs(curr_dist-bond_dists['C-O'])
    #    if curr_err < threshold:
    #        potential_O1_indexes.append(i)
    #
    #if len(potential_O1_indexes) != 1:
    #    print(potential_O1_indexes)
    #    print('ERROR FINDING O1 ATOM')
    #    return None
    #else:
    #    O1_index = potential_O1_indexes[0]    
    #INP_important_indexes = {'C1':C1_index,'C2':C2_index,'O1':O1_index}
    
    INP_important_indexes = {'C1':C1_index,'C2':C2_index}

    return INP_important_indexes   

def write_MM_orca_script(active_atoms_str,total_MM_charge,output_path):
    # Define the input and output file paths
    input_file = "scripts/template_MM_script.inp"
    # Open the input file and read its contents
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Replace the {} placeholders with the variable values
    content = content.replace("{}", "{" + active_atoms_str + "}" , 1)  # First occurrence
    # Add the custom line to the end
    custom_line = "*pdbfile " + str(total_MM_charge) +" 1 qm_complex.pdb"    
    content = content.replace("*pdbfile", custom_line , 1)  # First occurrence
    content += '\n'
    
    # Write the modified content to the output file
    file_name = "mm_opt.inp"
    output_file = output_path + file_name
    # Check if the directory exists
    if not os.path.exists(output_path):
        print(f"Directory '{output_path}' does not exist. Creating it...")
        os.makedirs(output_path)
    with open(output_file, 'w') as file:
        file.write(content)
    
    print(f"MM File processed and saved as {output_file}")

def write_QMMM_orca_script(QM_atoms_str,active_atoms_str,total_QM_charge,output_path):
    # Define the input and output file paths
    input_file = "scripts/template_QMMM_script.inp"    
    # Open the input file and read its contents
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Replace the {} placeholders with the variable values
    content = content.replace("{}", "{" + QM_atoms_str + "}" , 1)  # First occurrence
    content = content.replace("{}", "{" + active_atoms_str + "}", 1)  # Second occurrence
    # Add the custom line to the end
    custom_line = "*pdbfile " + str(total_QM_charge) +" 1 qm_complex.pdb\n"
    content += custom_line
    
    # Write the modified content to the output file
    file_name = "opt.inp"
    output_file = output_path + file_name
    # Check if the directory exists
    if not os.path.exists(output_path):
        print(f"Directory '{output_path}' does not exist. Creating it...")
        os.makedirs(output_path)
    with open(output_file, 'w') as file:
        file.write(content)
    print(f"Complex QM/MM File processed and saved as {output_file}")
    
def write_resids_to_csv(output_path,file_name,QM_residue_list,active_residue_list):
    
    output_file = output_path + file_name
    # Write to the CSV
    if not os.path.exists(output_path):
        print(f"Directory '{output_path}' does not exist. Creating it...")
        os.makedirs(output_path)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['QM RESIDUES'] + QM_residue_list)  # Write first list with label
        writer.writerow(['ACTIVE RESIDUES'] + active_residue_list)  # Write second list with label

    print(f"Data written to {output_file}")