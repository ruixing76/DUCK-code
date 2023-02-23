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
#for dataset in Twitter15 Twitter16 semeval2019
#do
#	if [ $dataset = semeval2019 ]
#	then
#		n_classes=3
#	else
#		n_classes=4
#	fi
#
#	folds=$(seq 0 4)
#	for fold in $folds
#	do
#		## Comment Tree
#		python train.py \
#			--datasetName $dataset \
#			--baseDirectory ./data \
#			--n_classes $n_classes \
#			--foldnum $fold \
#			--mode CommentTree \
#			--modelName Simple_GAT_BERT \
#			--batchsize $batch_size \
#			--learningRate 2e-5 \
#			--learningRateGraph 2e-5 \
#			--dropout_gat 0.2 \
#			--n_epochs 2
#		
#		## CCCT (Comment Chain + Comment Tree)
#		#python train.py \
#		#	--datasetName Twitter15 \
#		#	--baseDirectory ./data \
#		#	--n_classes 4 \
#		#	--foldnum 0 \
#		#	--mode CommentTree \
#		#	--modelName CCCTNet \
#		#	--batchsize 4
#	done
#done

## Hyperparameters tuning
for dataset in Twitter15 Twitter16 semeval2019
do
	if [ $dataset = semeval2019 ]
	then
		n_classes=3
	else
		n_classes=4
	fi

	fold=0
	for lr_bert in 1e-5 2e-5 3e-5 4e-5 5e-5
	do
		for lr_gnn in 1e-4 2e-4 3e-4 4e-4
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
				--dropout_gat 0.4 \
				--n_epochs 10
		done
	done
done