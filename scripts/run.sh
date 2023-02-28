#! /bin/bash

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	batch_size=2
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	batch_size=2
elif [ $(hostname) = "basic-1" ]; then
	export CUDA_VISIBLE_DEVICES=0
	batch_size=2
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	batch_size=4
fi

#Generate graph data for each dataset
#python vgae_preprocessing.py DATASET
#python vgae_preprocessing.py "Twitter15"

#Reproduce the experimental results
#python train.py MODEL_NAME DATASET

#for dataset in Twitter15
for dataset in Twitter15 Twitter16 semeval2019
do
	if [ $dataset = semeval2019 ]; then
		n_classes=3
	elif [ $dataset = Twitter15 ]; then
		n_classes=4
	elif [ $dataset = Twitter16 ]; then
		n_classes=4
	fi

	lr_bert=2e-5

	for lr_gnn in 1e-4 2e-4 4e-4
	do
		for dropout in 0.5 0.6
		do
			folds=$(seq 0 4)
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
					--learningRate $lr_bert \
					--learningRateGraph $lr_gnn \
					--dropout_gat $dropout \
					--n_epochs 20 \
					#--max_tree_len 40
			done
		done
	done
done

## Hyperparameters tuning
#for dataset in Twitter15 Twitter16 semeval2019
#do
#	if [ $dataset = semeval2019 ]
#	then
#		n_classes=3
#	else
#		n_classes=4
#	fi
#
#	fold=0
#	for lr_bert in 1e-5 2e-5 3e-5 4e-5 5e-5
#	do
#		for lr_gnn in 1e-4 2e-4 3e-4 4e-4
#		do
#			for dropout_gat in 0.2 0.3 0.4 0.5 0.6
#			do
#				## Comment Tree (BERT+GAT)
#				python train.py \
#					--datasetName $dataset \
#					--baseDirectory ./data \
#					--n_classes $n_classes \
#					--foldnum $fold \
#					--mode CommentTree \
#					--modelName Simple_GAT_BERT \
#					--batchsize $batch_size \
#					--learningRate $lr_bert \
#					--learningRateGraph $lr_gnn \
#					--dropout_gat $dropout_gat \
#					--n_epochs 10
#			done
#		done
#	done
#done