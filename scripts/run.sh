#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

#Generate graph data for each dataset
#python vgae_preprocessing.py DATASET
#python vgae_preprocessing.py "Twitter15"

#Reproduce the experimental results
#python train.py MODEL_NAME DATASET

## Comment Tree
python train.py \
	--datasetName Twitter15 \
	--baseDirectory ./data \
	--n_classes 4 \
	--foldnum 0 \
	--mode CommentTree \
	--modelName Simple_GAT_BERT \
	--batchsize 4

## CCCT (Comment Chain + Comment Tree)
#python train.py \
#	--datasetName Twitter15 \
#	--baseDirectory ./data \
#	--n_classes 4 \
#	--foldnum 0 \
#	--mode CommentTree \
#	--modelName CCCTNet \
#	--batchsize 4