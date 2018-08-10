#!/bin/bash
set -x
set -e

# Vars
setname=TBD # set name
emset=glove # pretrain set name
emdim=TBD
folds=TBD
gpuunit=TBD
node=gpu
logname=${setname}_${emset}${emdim}
gname=TBD # partition name

# Paths
embeddingpath=embedding
traindir=modeling
modelpath=models
modelname=$modelpath/${setname}_${emset}${emdim}_icon
sets=${setname}_sets
kwd=

# Dirs
mkdir -p ${modelpath}
mkdir -p error
mkdir -p out

folds=$(($folds-1))

model1=basic
model2=nce

echo "TEST"
# Test it w unknown data
sbatch -p ${node} --gres ${gname} --array=0-${folds} $traindir/testing.sh ${model1} ${sets} ${embeddingpath}/${emset}_${emdim}${kwd} True ${embeddingpath}/${emset}_${emdim}_icons${kwd} ${modelname} ${emdim} ${gpuunit}  ${logname} ${traindir}
