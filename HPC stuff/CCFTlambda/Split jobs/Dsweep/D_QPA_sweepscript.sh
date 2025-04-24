#!/bin/bash
#PBS -m a
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=16GB
#PBS -o $VSC_DATA/Vic/Lambda_estimation/Sweep/Split_jobs/Dsweep/Output/Epsilon/QPA/stdout.$PBS_JOBID
#PBS -e $VSC_DATA/Vic/Lambda_estimation/Sweep/Split_jobs/Dsweep/Error/Epsilon/QPA/stderr.$PBS_JOBID
#

STARTDIR=$VSC_DATA/Vic/Lambda_estimation/Sweep/Split_jobs/Dsweep #$PBS_O_WORKDIR for Home Directory

module load Julia/1.10.4-linux-x86_64

export I_MPI_COMPATIBILITY=8

# Move to proper working directory
cd $PBS_O_WORKDIR

# Run simulation
echo "Job started at : "`date`
julia --project=. --threads=1 "D_QPA_epsilon_singlerun.jl" --D $D
echo "Job ended at : "`date`