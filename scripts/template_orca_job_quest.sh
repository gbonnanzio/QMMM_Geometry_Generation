#!/bin/bash
#SBATCH -A p30041  
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem-per-cpu=4G
#SBATCH -t 48:00:00
#SBATCH --job-name="i2-"
#SBATCH --output=opt.txt

module load mpi/openmpi-4.1.1-gcc.10.2.0

SCRATCH=/scratch/gbf4422/$SLURM_JOB_ID
echo $SCRATCH

mkdir -p $SCRATCH
cp $SLURM_SUBMIT_DIR/*.inp $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.pdb $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.prms $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.gbw $SCRATCH/
cp $SLURM_SUBMIT_DIR/*.hess $SCRATCH/

cd $SCRATCH/

/projects/p30041/gbf4422/packages/orca_6_0_0/orca opt.inp > $SLURM_SUBMIT_DIR/opt.out

cp $SCRATCH/*.hess $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.gbw $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.xyz $SLURM_SUBMIT_DIR/
cp $SCRATCH/*.pdb $SLURM_SUBMIT_DIR/