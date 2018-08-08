#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=pretrained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="out/pretrained_%A_%a_%j.out"
#SBATCH --error="error/pretrained_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

corpus=$1
path2embd=$2
outdir=$3
srcdir=$4

srun python3 $srcdir/pretrained_embd.py --embdpath ${path2embd} --corpus ${corpus} --fout ${outdir}

