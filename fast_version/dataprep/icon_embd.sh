#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=icon_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="out/icon_embed_%A_%a_%j.out"
#SBATCH --error="error/icon_embed_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

path2embd=$1
emdim=$2
xml=$3
icon_name=$4
icond=$5
srcdir=$6

srun python3 $srcdir/icon_embd.py --embdspath $path2embd --embdim $emdim --xmlpath $xml --ficoname $icon_name --fidict $icond
