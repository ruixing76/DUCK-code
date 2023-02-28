This repository is forked from the official implementation of DUCK. Since the original project contains plenty of errors and unspecified parameters, I modify the code and further state the dataset preparation more clearly so that this project can be run.

# Dependencies:
- Python 3.8.10
```
$ pip install transformers==4.2.1
$ pip install Cython
$ pip install scikit-learn==0.21.3
$ pip install networkx==3.0
$ pip install pyro-ppl==0.3.0
$ pip install numpy==1.24.2
$ pip install pandas==1.4.4
$ pip install matplotlib
$ pip install ipdb
```
Install pytorch and pytorch-geometric as follows.
```
## Env: NVIDIA GeForce GTX 1080
$ pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
$ pip install torch-sparse==0.6.11 -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
$ pip install torch-geometric==2.2.0

## Env: NVIDIA GeForce RTX 3090
$ pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
$ pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
$ pip install torch-geometric==2.2.0
```

# Dataset
All datasets are publicly accessible.

[Twitter15](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 )
[Twitter16](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 )
[CoAID](https://github.com/cuilimeng/CoAID) version 0.4
[WEIBO](https://alt.qcri.org/~wgao/data/rumdect.zip)

## Data crawling tool
[twarc](https://github.com/DocNow/twarc)

# Train the DUCK model

```
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'DUCK' --modelName 'DUCK'
```

## comment graph data

```
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'CommentTree' --modelName 'Simple_GAT_BERT'
```

## user graph data

```
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'UserTree' --modelName 'Simple_GAT_BERT'
```

## run script

```
$ sh run.sh
```

# publicaton
This is the source code for 
[DUCK: Rumour Detection on Social Media by Modelling User and Comment Propagation Networks](https://aclanthology.org/2022.naacl-main.364/)


If you find this code useful, please let us know and cite our paper.  
If you have any question, please contact Lin at: s3795533 at student dot rmit dot edu dot au.
