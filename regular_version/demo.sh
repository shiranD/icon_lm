#!/bin/bash
set -x
set -e

# Vars
setname=TBD # set name
demodir=../fast_version/modeling
emset=TBD # name
emdim= # dimension
foldchoice= # your choice of test fold

# Paths
modelpath=models
modelname=$modelpath/${setname}_${emset}${emdim}_icon
embeddingpath=embedding
sets=${setname}_sets
pdfs=
ls ${pdfs} > filelist # list of icon pdf files
kwd=

# For demoing make sure you are on a gpu node
# for example: srun --pty --partition gpu --gres gpu:p100:1 /bin/bash (for interactive shell)
python3 $demodir/display.py --data ${sets}/set_${foldchoice}/ --embd ${embeddingpath}/${emset}_${emdim}${kwd} --icon ${embeddingpath}/${emset}_${emdim}_icons${kwd} --iconD ${embeddingpath}/${emset}_${emdim}_iconD${kwd} --load ${modelname}_${foldchoice}.pt --emsize $emdim --aug --path2pdfs ${pdfs} --flist filelist

#python3 $demodir/demo_reg.py --data ${sets}/set_${foldchoice}/ --embd ${embeddingpath}/${emset}_${emdim}${kwd} --icon ${embeddingpath}/${emset}_${emdim}_icons${kwd} --iconD ${embeddingpath}/${emset}_${emdim}_iconD${kwd} --load ${modelname}_${foldchoice}.pt --emsize $emdim --aug
