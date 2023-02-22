## Build Datasets
#python preprocess.py --make_label --dataset Twitter15
#python preprocess.py --make_label --dataset Twitter16
#python preprocess.py --make_label --dataset semeval2019

#python preprocess.py --split_5_fold --dataset Twitter15
#python preprocess.py --split_5_fold --dataset Twitter16
#python preprocess.py --split_5_fold --dataset semeval2019

python preprocess.py --build_graph --dataset Twitter15
python preprocess.py --build_graph --dataset Twitter16
python preprocess.py --build_graph --dataset semeval2019