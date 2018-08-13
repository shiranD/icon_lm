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
aug=--aug # augment with icons

# Paths
embeddingpath=embedding
testdir=modeling
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
sbatch -p ${node} --gres ${gname} --array=0-${folds} $testdir/testing.sh ${model1} ${sets} ${embeddingpath}/${emset}_${emdim}${kwd} ${aug} ${embeddingpath}/${emset}_${emdim}_icons${kwd} ${modelname} ${emdim} ${gpuunit}  ${logname} ${testdir}
