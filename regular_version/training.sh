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
traindir=../fast_version/modeling
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
for i in $(seq 0 $folds)
do
    python3 ${traindir}/train.py --modeltype ${model1} --data ${sets}/set_${i}/ --epochs ${numepochs} --embd ${embeddingpath}/${emset}_${emdim}${kwd} --aug --icon ${embeddingpath}/${emset}_${emdim}_icons${kwd} --save ${modelname}_${i}.pt --emsize ${emdim} --trainlog out/trainlog_${logname}_${i}
done
