#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output="out/trainshort_%A_%a_%j.out"
#SBATCH --error="error/trainshort_%A_%a_%j.err"
### -------------------------- ###
echo "job name: $SLURM_JOB_NAME"
echo "SLURM_JOBID:  $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/lib
export CFLAGS=-I/usr/local/cuda-8.0/include
export LDFLAGS=-L/usr/local/cuda-8.0/lib64
export PATH=$PATH:/usr/local/cuda-8.0/bin
export CUDA_HOME=/usr/local/cuda-8.0
export LIBRARY_PATH=/usr/local/cuda-8.0/lib64

model=$1
data=$2
epoch=$3
embdpath=$4
aug=$5
iconpath=$6
modelname=$7
dim=$8
logname=$9
srcdir=${10}
if [ "${aug}"='True' ]; then
  aug=--aug
  echo yes
else
  aug=''
fi;

python3 ${srcdir}/train.py --modeltype ${model} --data ${data}/set_${SLURM_ARRAY_TASK_ID}/ --epochs ${epoch} --embd ${embdpath} ${aug} --icon ${iconpath} --save ${modelname}_${SLURM_ARRAY_TASK_ID}.pt --emsize ${dim} --cuda --trainlog out/trainlog_${logname}_${SLURM_ARRAY_TASK_ID}
