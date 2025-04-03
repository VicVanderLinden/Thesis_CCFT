#!/bin/bash
#
#PBS -N Vic lambda estimation
#PBS -m a
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=64GB
#

STARTDIR=$VSC_DATA/Vic/Lambda_estimation #$PBS_O_WORKDIR for Home Directory

module load Julia/1.10.4-linux-x86_64 

cd $STARTDIR
echo "PBS: $PBS_ID"

ls

echo "Job started at : "`date`
julia --project=. --threads=1 "lambda_estimation.jl"
echo "Job ended at : "`date`