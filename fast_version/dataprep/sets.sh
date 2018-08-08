#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=sets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="out/sets_%A_%a_%j.out"
#SBATCH --error="error/sets_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

folds=$1
sets=$2
num=$3
srcdir=$4

mkdir -p $sets/set_$SLURM_ARRAY_TASK_ID
num=$(($num+1))
srun python3 $srcdir/sets.py --foldspath $folds --path2sets $sets/set_$SLURM_ARRAY_TASK_ID/ --foldnum $SLURM_ARRAY_TASK_ID --numfolds $num
