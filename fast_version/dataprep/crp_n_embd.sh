#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=crp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output="out/crp_%A_%a_%j.out"
#SBATCH --error="error/crp_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

path2corpus=$1
path2embd=$2
new_corpus=$3
srcdir=$4
srun python3 ${srcdir}/crp_n_embd.py --fdata ${path2corpus}_$SLURM_ARRAY_TASK_ID --fembd ${path2embd} > ${new_corpus}_${SLURM_ARRAY_TASK_ID}_a
awk 'NF' ${new_corpus}_${SLURM_ARRAY_TASK_ID}_a > ${new_corpus}_${SLURM_ARRAY_TASK_ID}
rm ${new_corpus}_${SLURM_ARRAY_TASK_ID}_a
