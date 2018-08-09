#!/bin/bash
set -x
set -e

# Vars
emset=TBD # name (or type) of pretrained file (not path sensitive)
emdim=TBD # dimentions of embeddings
folds=TBD # how many folds
corpusname=NAME # corpus name

# Paths
path2corpus=TBD
path2trained_embd=TBD
xmlpath=TBD
new_embd_dir=embedding
dataf=data # data folder name
final_corpus=${dataf}/corpus_CE
mid_corpus=${dataf}/corpus_ICE
start_corpus=${dataf}/corpus
srcdir=../fast_version/dataprep
traindir=modeling
kwd= # (optional) additinal keyword to differentiate similar trainings

# mk Dirs
mkdir -p ${dataf}
mkdir -p ${traindir}
mkdir -p ${start_corpus}
mkdir -p ${new_embd_dir}
mkdir -p ${final_corpus}
mkdir -p ${dataf}/folds
mkdir -p ${dataf}/sets
mkdir -p ${mid_corpus}


echo "PRE PROCESS DATA"

# optional: prior to this process you may train you own embedding to represent the icons.

# generate icon embeddings from pretrained embeddings
python3 $srcdir/icon_embd.py --embdspath ${path2trained_embd} --embdim ${emdim} --xmlpath ${xmlpath} --ficoname ${new_embd_dir}/${emset}_${emdim}_icons${kwd} --fidict ${new_embd_dir}/${emset}_${emdim}_iconD${kwd}

# iconize the corpus. Replace a term with an icon id
python3 $srcdir/iconize_corpus.py --fdata ${start_corpus}/${corpusname} --fembd ${new_embd_dir}/${emset}_${emdim}_iconD${kwd} > ${mid_corpus}/${corpusname}

# replace non-icon terms with pretrained terms, and the rest with an <unk> symobol
python3 ${srcdir}/crp_n_embd.py --fdata ${mid_corpus}/${corpusname} --fembd ${path2trained_embd} > ${final_corpus}/${corpusname}_a
awk 'NF' ${final_corpus}/${corpusname}_a > ${final_corpus}/${corpusname}
rm ${final_corpus}/${corpusname}_a

# split file N ways
python3 ${srcdir}/split.py --fdata ${final_corpus}/${corpusname} --folds ${folds} --odir ${dataf}/folds/fold

# create sets
folds=$(($folds-1))
for i in $(seq 1 ${folds})
do
  mkdir ${dataf}/sets_${i}
done
python3 $srcdir/sets_serial.py --foldspath ${dataf}/folds/fold --path2sets $sets/set_ --numfolds ${folds}

# extract the relevant pretrained embeddings for the corpus (reduce memory) can be done w icon embedding too
python3 $srcdir/pretrained_embd.py --embdpath ${path2trained_embd} --corpus ${final_corpus}/${corpusname} --fout ${new_embd_dir}/${emset}_${emdim}${kwd}
