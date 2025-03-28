# %%
import MDAnalysis as mda
import numpy as np
import os
import shutil
import csv
import networkx as nx
from utils import *
import warnings 

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# dictionary of charged protein residues with key being the Amber resname and the value being the net charge of that residue
residue_charge_dict = {'ARG':1,'LYS':1,'ASP':-1,'GLU':-1,'MG':2,'ThDP':-3}

'''
The purpose of this script is to read in the end point of an int1 to DMC (Donor Michaelis-Complex) scan and remake the MM parameters for the 
unbound donor. This way we can follow the scan with an MM minimization where the donor can move and the ThDP is kept fixed. This script will
identify what atoms in INI (intermediate 1) are the donor and what atoms are ThDP. It will then rename the residues and call a bash script 
to run the AMBER commands to get the new set of parameters. It will also create the MM optimization script.
'''

head_dir = 'temp_structures/'
curr_substrate= '6'

substrate_base_charge = -1  
starting_intermediate = 'INI'
ending_intermediate = 'DON'

working_dir = head_dir 
# read in endpoint of the scan 
donor_substrate = curr_substrate
complex = mda.Universe(working_dir + '/tmp.pdb')
QM_resids = [1113,1114]
MM_resids = [30,31,32,33,34,78,79,80,82,107,117,118,175,946,947,948,950,951,969,971,972,974,999,1000,1003,1028,1030,1031,1034,1045,1113,1114,1225,1919,2292,2529,2894,3100,3475,3645,3670,3953,4130,4203,4248,4262,4328,4363,4548,5237,5303,5351,5422] #read_resids_from_csv(working_dir + 'prep/QM_2_and_10A_Active_residues.csv') 

# Get the charge of the QM and MM systems 
if curr_substrate in ['4','6','11','16']:
    additional_charge = -1
else:
    additional_charge = 0

residue_charge_dict[ending_intermediate] = substrate_base_charge + additional_charge

# get the complex atoms 
ini = complex.select_atoms("resname " + starting_intermediate)

# guess bonds based on vDW distances (this is not recommended)
ini_bond_connectivity = mda.topology.core.guess_bonds(ini.atoms,ini.positions)

# Create a graph where nodes are atom indices and edges are bonds
ini_graph = nx.Graph()
ini_graph.add_edges_from(ini_bond_connectivity)

# Find connected components
connected_components = list(nx.connected_components(ini_graph))
connected_components = sorted(connected_components,key=len)
donor_indexes = list(connected_components[0])
ThDP_indexes = list(connected_components[1])

# redesignate donor and ThDP indexes
donor = complex.select_atoms(f"index {' or index '.join(map(str, donor_indexes))}")
for atom in donor.atoms:
    atom.residue.resid = 1
    atom.residue.resname = "DON"
    atom.record_type = "HETATM"
write_universe("temp_structures/","donor.pdb",donor)
ThDP = complex.select_atoms(f"index {' or index '.join(map(str, ThDP_indexes))}")
for atom in ThDP.atoms:
    atom.residue.resid = 1
    atom.residue.resname = "TPP"
    atom.record_type = "HETATM"
write_universe("temp_structures/","TPP.pdb",ThDP)

