#!/bin/bash
#SBATCH -A TG-CTS120055
#SBATCH --partition=shared
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem-per-cpu=4G
#SBATCH -t 48:00:00 
#SBATCH --job-name=""
#SBATCH --output=opt.txt

module reset

module load cpu/0.17.3b
module load gcc/10.2.0
module load openmpi/4.1.1

SCRATCH=/expanse/lustre/scratch/gbonnanzio/temp_project/$SLURM_JOB_ID
echo $SCRATCH

mkdir -p $SCRATCH
cp $SLURM_SUBMIT_DIR/*.inp $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.pdb $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.prms $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.gbw $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.hess $SCRATCH/

cd $SCRATCH/

/expanse/lustre/projects/nwu118/gbonnanzio/packages/orca_6_0_0/orca opt.inp > $SLURM_SUBMIT_DIR/opt.out

cp $SCRATCH/*.hess $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.gbw $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.xyz $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.pdb $SLURM_SUBMIT_DIR/
