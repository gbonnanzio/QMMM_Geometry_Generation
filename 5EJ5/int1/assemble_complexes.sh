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
dirs=("14" "2" "5" "6" "11" "18")

# Iterate through all directories in the current directory
for dir in "${dirs[@]}"; do
    cd "$dir"
    echo "$dir"
    # Get the base directory
    last_dir_name=$(basename "$dir")
    if [[ "$last_dir_name" == 4 || "$last_dir_name" == 6 || "$last_dir_name" == 11 || "$last_dir_name" == 16 ]]; then
        charge="-5"
    else
        charge="-4"
    fi
    pdb4amber -i ini.pdb -o ini_amber.pdb 
    antechamber -i ini_amber.pdb -o INI.mol2 -fi pdb -fo mol2 -c bcc -pf yes -nc $charge -at gaff2 -j 5
    parmchk2 -i INI.mol2 -f mol2 -o INI.frcmod -s 2 

    pdb4amber -i receptor.pdb -o receptor_amber.pdb
    pdb4amber -i water.pdb -o water_amber.pdb

    tleap -f ../tleap_script.in
    orca_mm -convff -amber qm_complex.prmtop
    
    cd ..

     
done
