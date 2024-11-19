#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=10G
#SBATCH -t 01:00:00
#SBATCH --job-name="system_prep"
#SBATCH --output=system_prep.txt

module load ambertools
module load orca

# Get the current directory
current_dir=$(pwd)

# Iterate through all directories in the current directory
for dir in "$current_dir"/*/; do
    cd "$dir"
    echo "$dir"
    # Get the base directory
    last_dir_name=$(basename "$dir")
    if [[ "$last_dir_name" == 4 || "$last_dir_name" == 6 || "$last_dir_name" == 11 || "$last_dir_name" == 16 ]]; then
        charge="-5"
    else
        charge="-4"
    fi

    #pdb4amber -i ini.pdb -o ini_amber.pdb 
    #antechamber -i ini_amber.pdb -o INI.mol2 -fi pdb -fo mol2 -c bcc -pf yes -nc $charge -at gaff2 -j 5
    #parmchk2 -i INI.mol2 -f mol2 -o INI.frcmod -s 2 

    pdb4amber -i receptor.pdb -o receptor_amber.pdb
    pdb4amber -i water.pdb -o water_amber.pdb
    
    tleap -f ../tleap_script.in

    orca_mm -convff -amber qm_complex.prmtop
    
done
