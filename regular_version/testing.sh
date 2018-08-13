#!/bin/bash
set -x
set -e

# Vars
setname=TBD # set name
emset=TBD # pretrain set name
emdim=TBD # vector dimesions
folds=TBD # number of folds
logname=${setname}_${emset}${emdim}
aug=--aug # augment eith icons

# Paths
embeddingpath=embedding # path to embedding folder
testdir=../fast_version/modeling # path to test folder
modelpath=models # path to models' folder
modelname=$modelpath/${setname}_${emset}${emdim}_icon
sets=${setname}_sets
kwd=

folds=$(($folds-1))

model1=basic
model2=nce

echo "TEST"
for i in $(seq 0 $folds)
do
    # Test it w unknown data
    python3 ${testdir}/testing_reg.py --modeltype ${model2} --data ${sets}/set_${i}/ --embd ${embeddingpath}/${emset}_${emdim}${kwd} ${aug} --icon ${embeddingpath}/${emset}_${emdim}_icons${kwd} --load ${modelname}_${i}.pt --emsize ${emdim} --testlog out/testlog_${logname}_${i} --fold ${i}
done
