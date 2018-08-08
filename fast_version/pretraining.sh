#!/bin/bash
set -x
set -e
# Vars
emset=TBD # name (or type) of pretrained file (not path sensitive)
emdim=TBD # dimentions of embeddings
folds=TBD # how many folds
corpusname=NAME # corpus name

path2corpus=TBD
path2trained_embd=TBD
xmlpath=TBD
new_embd_dir=embedding
dataf=data # data folder name
# Paths
#xmlpath=TBD # to retrieve icons from
#new_embd_dir=TBD # embedding foldername
#path2corpus=TBD # 
#path2trained_embd=${new_embd_dir}/${emset}_${emdim} # this is the file name location of the embedding
final_corpus=${dataf}/corpus_CE
mid_corpus=${dataf}/corpus_ICE
start_corpus=${dataf}/corpus
srcdir=dataprep
traindir=modeling
kwd= # (optional) additinal keyword to differentiate similar trainings

# mk Dirs
mkdir -p ${dataf}
mkdir -p ${traindir}
mkdir -p ${start_corpus}
mkdir -p ${new_embd_dir}
mkdir -p ${final_corpus}
mkdir -p error
mkdir -p out
mkdir -p ${dataf}/folds
mkdir -p ${dataf}/sets
mkdir -p ${mid_corpus}


echo "PRE PROCESS DATA"

# train embeddings from subtlex (optional)
#pass
# FIND SENTENCES of the corpus that can be represented purely by EMBEDDINGS

# 1) split corpus (-l 1000 should vary given the size of your corpus and node capacity)
split -l 1000 -d -a 3 ${path2corpus} ${start_corpus}/${corpusname}_ 
i=0
for FILE in `ls ${start_corpus}/`
 do  
 mv ${start_corpus}/$FILE ${start_corpus}/${corpusname}_$i  
 let i=i+1
done

# count number of files
num=$(ls -l ${dataf}/corpus | grep ${corpusname} | wc -l)
num=$(($num-1))

# generate icon embeddings from pretrained embeddings
FIRST=$(sbatch $srcdir/icon_embd.sh ${path2trained_embd} $emdim $xmlpath ${new_embd_dir}/${emset}_${emdim}_icons${kwd} ${new_embd_dir}/${emset}_${emdim}_iconD${kwd}  $srcdir | cut -f4 -d' ')

# iconize the corpus. Replace a term with an icon id
SECOND=$(sbatch --dependency=afterany:${FIRST} --array=0-$num ${srcdir}/iconize_corpus.sh ${start_corpus}/${corpusname} ${new_embd_dir}/${emset}_${emdim}_iconD${kwd} ${mid_corpus}/${corpusname} $srcdir | cut -f 4 -d' ' ) 

# replace non-icon terms with pretrained terms, and the rest with an <unk> symobol
THIRD=$(sbatch --dependency=afterany:${SECOND} --array=0-$num ${srcdir}/crp_n_embd.sh ${mid_corpus}/${corpusname} ${path2trained_embd} ${final_corpus}/${corpusname} ${srcdir} | cut -f 4 -d' ' )

# SPLIT DATA
# merge subfiles to one
# split file N ways
folds=$(($folds-1))
FOURTH=$(sbatch --dependency=afterany:${THIRD} $srcdir/split.sh ${final_corpus}/${corpusname} $folds ${dataf}/folds/fold $num $srcdir | cut -f 4 -d' ' )

# create sets
FIFTH=$(sbatch --dependency=afterany:${FOURTH} --array=0-$folds $srcdir/sets.sh ${dataf}/folds/fold ${dataf}/sets $folds $srcdir | cut -f 4 -d' ' )

# extract the relevant pretrained embeddings for the corpus (reduce memory) can be done w icon embedding too
SIXTH=$(sbatch --dependency=afterany:${FIFTH} $srcdir/pretrained.sh ${final_corpus}/${corpusname} ${path2trained_embd} ${new_embd_dir}/${emset}_${emdim}${kwd}  $srcdir | cut -f 4 -d' ' )
