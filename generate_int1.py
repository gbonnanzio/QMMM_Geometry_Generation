import MDAnalysis as mda
import numpy as np
import os
from utils import *

# read in each of the substrate files
directory = 'substrates_initial/'
file_names = [f for f in os.listdir(directory) if "_head" not in f]

for curr_file_name in file_names:
    file_start = curr_file_name.split('.')[0]
    # load substrate universe
    substrate = mda.Universe(directory+curr_file_name)
    # identify the atoms that comprise the aka substrates 
    substrate_important_indexes = get_substrate_aka_indexes(substrate)

    receptor = mda.Universe('int1_receptor.pdb')
    ThDP_important_indexes = get_ThDP_indexes(receptor)

    tpp_residue = receptor.select_atoms("resname TPP")

    C1_coords = get_atom_position(tpp_residue,ThDP_important_indexes['C1'])
    N1_coords = get_atom_position(tpp_residue,ThDP_important_indexes['N1'])
    N2_coords = get_atom_position(tpp_residue,ThDP_important_indexes['N2'])
    S1_coords = get_atom_position(tpp_residue,ThDP_important_indexes['S1'])

    vector_S1_to_C1 = C1_coords - S1_coords
    vector_N1_to_C1 = C1_coords - N1_coords
    avg_vector = (vector_S1_to_C1 + vector_N1_to_C1)/2
    unit_vector = avg_vector / np.linalg.norm(avg_vector)
    guess_C2 = C1_coords + unit_vector * 1.54

    # Example input
    # ThDP C1, N1, N2, S1 atom coords
    centers = np.array([C1_coords,N1_coords,N2_coords,S1_coords])
    # radii are in order of (columns) C1, N1, N2, S1 and then rows (C2, C3, O1) 
    radii = [
        [1.539,2.562,3.389,2.880],
        [2.533,3.205,4.764,3.784],
        [2.393,2.973,2.592,3.893]
    ]
    #initial_guess = np.hstack([np.mean(centers, axis=0) for i in range(3)])
    initial_guess = np.hstack([guess_C2 for i in range(3)])
    #print(init)
    C2_optimized, C3_optimized, O1_optimized = optimize_points(centers, initial_guess, radii)
    all_optimized = [C2_optimized, C3_optimized, O1_optimized]
    print("Optimized points:", C2_optimized, C3_optimized, O1_optimized)

    C2_err = atom_objective(C2_optimized, centers, radii[0])
    C3_err = atom_objective(C3_optimized, centers, radii[1])
    O1_err = atom_objective(O1_optimized, centers, radii[2])

    all_errors = [C2_err,C3_err,O1_err]
    min_error_index = all_errors.index(min(all_errors))
    redo_initial_guess = np.hstack([all_optimized[min_error_index] for i in range(3)])

    C2_reoptimized, C3_reoptimized, O1_reoptimized = optimize_points(centers, redo_initial_guess, radii)
    print("Reoptimized points:", C2_reoptimized, C3_reoptimized, O1_reoptimized)

    initial_positions = [get_atom_position(substrate,substrate_important_indexes['C2']),get_atom_position(substrate,substrate_important_indexes['C3']),get_atom_position(substrate,substrate_important_indexes['O1'])]
    final_positions = [C2_reoptimized, C3_reoptimized, O1_reoptimized]

    R, t = kabsch_algorithm(initial_positions,final_positions)

    for i in range(0,len(substrate.atoms.positions)):
        atom_coords = substrate.atoms[i].position
        new_coords = np.dot(R, atom_coords) + t
        substrate.atoms[i].position = new_coords

    C2_coords = substrate.atoms.positions[substrate_important_indexes['C2']]
    O1_coords = substrate.atoms.positions[substrate_important_indexes['O1']]
    C3_coords = substrate.atoms.positions[substrate_important_indexes['C3']]
    R_coords =  substrate.atoms.positions[substrate_important_indexes['R']]

    target_angles = [111.1,110.2,107.5]
    R_coord_guess = optimize_angles(R_coords,C1_coords,C2_coords,O1_coords,C3_coords,target_angles)

    t_R = R_coord_guess-R_coords

    substrate_tail_atom_indexes = [i for i in range(0,len(substrate.atoms)) if i not in substrate_important_indexes.values()]
    substrate_tail_atom_indexes.append(substrate_important_indexes['R'])


    for i in range(0,len(substrate.atoms)):
        if i in substrate_tail_atom_indexes:
            atom_coords= substrate.atoms.positions[i]
            new_coords = atom_coords + t_R
            substrate.atoms[i].position = new_coords

    # Save the updated universe to a new PDB file
    output_filename = directory + file_start +'_int1.pdb'
    substrate.atoms.write(output_filename)

