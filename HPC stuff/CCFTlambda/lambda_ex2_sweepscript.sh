#!/bin/bash
#PBS -m a
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=8GB
#PBS -o $VSC_DATA/Vic/Lambda_estimation/Sweep/Output/stdout.$PBS_JOBID
#PBS -e $VSC_DATA/Vic/Lambda_estimation/Sweep/Error/EXC2/stderr.$PBS_JOBID
#

STARTDIR=$VSC_DATA/Vic/Lambda_estimation/Sweep #$PBS_O_WORKDIR for Home Directory

module load Julia/1.10.4-linux-x86_64

export I_MPI_COMPATIBILITY=8

# Move to proper working directory
cd $PBS_O_WORKDIR

# Run simulation
echo "Job started at : "`date`
julia --project=. --threads=1 "ex2_singlerun.jl" --lambdar $lambdar --lambdaim $lambdaim --L $L
echo "Job ended at : "`date`