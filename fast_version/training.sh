#!/bin/bash
set -x
set -e

# Vars
setname=TBD #name
emset=TBD # pretrained set name 
emdim=TBD # embedding dimension
folds=TBD
numepochs=40
sets=${setname}_sets
dataf=data
traindir=modeling
node=gpu
gname= # the gpu partition
logname=${setname}_${emset}${emdim}
model1=basic
model2=nce
kwd= # optional

# Paths
embeddingpath=embedding # path to embedidng folder
modelpath=models # path to future models
modelname=$modelpath/${setname}_${emset}${emdim}_icon

# Dirs
if [ ! -d ${dataf}/sets ${sets} ]; then
  mv ${dataf}/sets ${sets}
fi
mkdir -p ${modelpath}
mkdir -p error
mkdir -p out

folds=$(($folds-1))

echo "TRAIN"
sbatch -p ${node} --gres ${gname} --array=0-$folds $traindir/train.sh ${model2} ${sets} ${numepochs} ${embeddingpath}/${emset}_${emdim}${kwd} True ${embeddingpath}/${emset}_${emdim}_icons${kwd} ${modelname} ${emdim} ${logname} ${traindir}
