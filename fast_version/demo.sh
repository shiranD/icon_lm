#!/bin/bash
set -x
set -e

# Vars
setname=TBD # set name
demodir=modeling
emset=TBD # name
emdim=TBD
foldchoice=TBD

# Paths
modelpath=models
modelname=$modelpath/${setname}_${emset}${emdim}_icon
embeddingpath=embedding
sets=${setname}_sets
kwd=

# For demoing make sure you are on a gpu node
# for example: srun --pty --partition gpu --gres gpu:p100:1 /bin/bash (for interactive shell)

python3 $demodir/demo.py --data ${sets}/set_${foldchoice}/ --embd ${embeddingpath}/${emset}_${emdim}${kwd} --icon ${embeddingpath}/${emset}_${emdim}_icons${kwd} --iconD ${embeddingpath}/${emset}_${emdim}_iconD${kwd} --load ${modelname}_${foldchoice}.pt --emsize $emdim --cuda --aug
