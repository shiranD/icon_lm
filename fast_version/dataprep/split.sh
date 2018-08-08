#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="out/split_%A_%a_%j.out"
#SBATCH --error="error/split_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

fnames=$1
numfolds=$2
foldpath=$3
srcdir=$4

cat ${fnames}_* > ${fnames}
numfolds=$(($numfolds+1))
python3 ${srcdir}/split.py --fdata ${fnames} --folds ${numfolds} --odir ${foldpath}

