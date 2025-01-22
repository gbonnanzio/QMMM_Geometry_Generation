import MDAnalysis as mda
import numpy as np
import os
import shutil
import csv
from utils import *
import warnings 

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# load receptor universe and extract the different parts of the protein int1 receptor 
head_dir = '/projects/p30041/gbf4422/5EJ5/int3_R/6/'
complex = mda.Universe(head_dir+'run/last_equil_frame.pdb')

output_dir = head_dir + 'QMMM/'

# get all atoms near the protein excluding NaCl
trimmed_complex= complex.select_atoms("(protein or around 3 protein) and (protein or resname INP or resname MG or resname WAT)").residues

write_universe(output_dir  + 'int3/prep/','initial.pdb',trimmed_complex)
edit_protein_files(output_dir  + 'int3/prep/','initial.pdb')

trimmed_complex = mda.Universe(output_dir  + 'int3/prep/initial.pdb')
# dictionary of charged protein residues with key being the Amber resname and the value being the net charge of that residue
residue_charge_dict = {'ARG':1,'LYS':1,'HIP':1,'ASP':-1,'GLU':-1,'MG':2}

donor_substrate = '6'
acceptor_substrate = '6'

# specify active atoms
by_dist = True 
QM_sphere_r = 2 # Angstroms (a value of 2 will give only the ThDP intermediate (auto adds MG))
active_sphere_r = 10
    
# TODO calculate protein charge automatically (or read it in from leap.log file)
base_charge = -4 # inp without accounting for R groups, you will never need to change this for INP      
intermediate = 'INP'


if acceptor_substrate in ['4','6','11','16']:
    additional_charge = -1
else:
    additional_charge = 0
# add the charge of the intermediate complex
if donor_substrate in ['4','6','11','16']:
    additional_charge += -1



residue_charge_dict[intermediate] = base_charge + additional_charge
MM_charge = 0
for residue in trimmed_complex.residues:
    curr_resname = residue.resname
    if curr_resname in residue_charge_dict:
        MM_charge += residue_charge_dict[curr_resname]

# get the QM atoms and residues (automatically includes Mg2+)
# get INP atoms 
inp = trimmed_complex.select_atoms("resname " + intermediate)
# get the carbonyl carbon of the substrate
#atom_dict = get_substrate_aka_indexes(ini)
atom_dict = get_inp_indexes(inp)
C2_index = atom_dict['C2']
C2_id = inp.atoms[C2_index].index
C2_atom = trimmed_complex.select_atoms("index " +  str(C2_id))
QM_residues, active_residues, QM_atoms_indexes,active_atoms_indexes,fixed_atoms_indexes = get_atoms_by_distance(trimmed_complex,QM_sphere_r,active_sphere_r,C2_id)
QM_residues_resids = [residue.resid for residue in QM_residues]
active_residues_resids = [residue.resid for residue in active_residues]

# simplify lists to write to file
QM_list = simplify_integer_list(QM_atoms_indexes)
active_list = simplify_integer_list(active_atoms_indexes)
# calculate the charge of our system
total_QM_charge = 0
# get charge of QM region 
for residue in QM_residues:
    resname = residue.resname
    if resname in residue_charge_dict:
        total_QM_charge += residue_charge_dict[resname]

#qm_complex_output_dir = output_dir + 'QM_' + str(QM_sphere_r) + '_Active_' + str(active_sphere_r) + '/'
#write_MM_orca_script(active_list,MM_charge,output_dir+'MM_Opt_Active_10/')
shutil.copy(head_dir + 'prep/INP.mol2', output_dir + 'int3/prep/INP.mol2')
shutil.copy(head_dir + 'prep/INP.frcmod', output_dir + 'int3/prep/INP.frcmod')


write_QMMM_orca_script(QM_list,active_list,total_QM_charge,output_dir + 'int3/run/')
#shutil.copy(qm_complex_file_dir + '/qm_complex.ORCAFF.prms', qm_complex_output_dir + 'qm_complex.ORCAFF.prms')
#shutil.copy(qm_complex_file_dir + '/qm_complex.pdb', qm_complex_output_dir + 'qm_complex.pdb')
#shutil.copy('scripts/template_orca_job_expanse.sh', qm_complex_output_dir + 'orca_job_expanse.sh')
#write_resids_to_csv(qm_complex_output_dir,f'QM_{QM_sphere_r}_and_{active_sphere_r}A_Active_residues.csv',QM_residues_resids,active_residues_resids)


