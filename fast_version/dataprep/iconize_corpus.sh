#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=iconize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="out/iconize_%A_%a_%j.out"
#SBATCH --error="error/iconize_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

corpus=$1
embedding=$2
embedded_corpus=$3
srcdir=$4

python3 $srcdir/iconize_corpus.py --fdata ${corpus}_$SLURM_ARRAY_TASK_ID --fembd ${embedding} > ${embedded_corpus}_$SLURM_ARRAY_TASK_ID
