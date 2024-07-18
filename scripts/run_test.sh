#! /bin/bash


export CUDA_VISIBLE_DEVICES=0
batch_size=4


#Generate graph data for each dataset
#python vgae_preprocessing.py DATASET
#python vgae_preprocessing.py "Twitter15"

#Reproduce the experimental results
#python train.py MODEL_NAME DATASET

#for dataset in Twitter15
for dataset in Twitter15
do
	if [ $dataset = semeval2019 ]
	then
		n_classes=3
	else
		n_classes=4
	fi

	folds=$(seq 0 1)
	for fold in $folds
	do
		## Comment Tree
		python train.py \
			--datasetName $dataset \
			--baseDirectory ./data \
			--n_classes $n_classes \
			--foldnum $fold \
			--mode CommentTree \
			--modelName Simple_GAT_BERT \
			--batchsize $batch_size \
			--learningRate 2e-5 \
			--learningRateGraph 3e-4 \
			--dropout_gat 0.5 \
			--n_epochs 10 \
			--max_tree_len 40

	done
done
