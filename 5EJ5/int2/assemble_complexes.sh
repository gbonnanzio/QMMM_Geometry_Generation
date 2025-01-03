#!/bin/bash
#SBATCH -A TG-CTS120055
#SBATCH --partition=shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=10G
#SBATCH -t 01:00:00 
#SBATCH --job-name="prep"
#SBATCH --output=run.txt

module purge
module load cpu/0.17.3b
module load gcc/10.2.0
module load openmpi/4.1.1
module load orca/5.0.4

conda init bash 
source ~/.bashrc
conda activate /home/gbonnanzio/.conda/envs/ambertools-env


# Get the current directory
current_dir=$(pwd)


dirs=("0" "2" "4" "6" "11" "14")

# Iterate through all directories in the current directory
for dir in "${dirs[@]}"; do
    cd "$dir"
    echo "$dir"
    # Get the base directory
    last_dir_name=$(basename "$dir")
    if [[ "$last_dir_name" == 4 || "$last_dir_name" == 6 || "$last_dir_name" == 11 || "$last_dir_name" == 16 ]]; then
        charge="-4"
    else
        charge="-3"
    fi
    pdb4amber -i ino.pdb -o ino_amber.pdb 
    antechamber -i ino_amber.pdb -o INO.mol2 -fi pdb -fo mol2 -c bcc -pf yes -nc $charge -at gaff2 -j 5
    parmchk2 -i INO.mol2 -f mol2 -o INO.frcmod -s 2 

    pdb4amber -i receptor.pdb -o receptor_amber.pdb
    #pdb4amber -i water.pdb -o water_amber.pdb

    tleap -f ../tleap_script.in
    orca_mm -convff -amber qm_complex.prmtop
    
    cd ..
     
done
