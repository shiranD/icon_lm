#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output="out/testing_%A_%a_%j.out"
#SBATCH --error="error/testing_%A_%a_%j.err"
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
embdpath=$3
aug=$4
iconpath=$5
modelname=$6
dim=$7
gpu=$8
if [ "${aug}"='True' ]; then
  aug=--aug
  echo yes
else
  aug=''
fi;
logname=$9
srcdir=${10}

python3 ${srcdir}/testing.py --modeltype ${model} --data ${data}/set_${SLURM_ARRAY_TASK_ID}/ --embd ${embdpath} ${aug} --icon ${iconpath} --save ${modelname}_${SLURM_ARRAY_TASK_ID}.pt --emsize ${dim} --cuda --trainlog out/testlog_${logname}_${SLURM_ARRAY_TASK_ID} --gpu $gpu --fold ${SLURM_ARRAY_TASK_ID} 
