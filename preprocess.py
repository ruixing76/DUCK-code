import os
import csv
import ipdb
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing for DUCK")

    ## What to do
    parser.add_argument("--make_label", action="store_true")
    parser.add_argument("--build_graph", action="store_true")
    parser.add_argument("--split_5_fold", action="store_true")

    ## Others
    #parser.add_argument("--data_root", type=str, default="../dataset/processed")
    #parser.add_argument("--data_root_V2", type=str, default="../dataset/processedV2")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4")
    parser.add_argument("--data_root", type=str, default="./data", help="root directory for DUCK's data")
    parser.add_argument("--data_source", type=str, default="./dataset")
    parser.add_argument("--dataset", type=str, default="Twitter15", choices=["semeval2019", "Twitter15", "Twitter16"])

    args = parser.parse_args()

    return args


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq = float(pair.split(':')[1])
        index = int(pair.split(':')[0])
        if index <= 5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex


def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex = indexC - 1
            root_index = nodeC.index
            root_word = nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index) > 0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_word = []
    x_index = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children != None and index2node[index_j + 1] in index2node[index_i + 1].children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i + 1].word)
        x_index.append(index2node[index_i + 1].index)
    edgematrix = [row, col]
    return x_word, x_index, edgematrix, rootfeat, rootindex


def getfeature(x_word, x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i]) > 0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x


def constructMat_txt(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        text = tree[j]['vec']  # raw text
        nodeC.text = text
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex = indexC - 1
            root_text = text
    matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_text = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i + 1].children != None and index2node[index_j + 1] in index2node[index_i + 1].children:
                matrix[index_i][index_j] = 1
                row.append(index_i)
                col.append(index_j)
        #x_text.append(clean_data(index2node[index_i+1].text))
        x_text.append(index2node[index_i + 1].text)
    if row == [] and col == []:
        matrix[0][0] = 1
        row.append(0)
        col.append(0)
    edgematrix = [row, col]
    return x_text, edgematrix, root_text, rootindex


def build_graph(args):
    treePath = "{}/{}/data.csv".format(args.data_source, args.dataset.lower())
    savePath = "{}/{}graph".format(args.data_root, args.dataset)
    os.makedirs(savePath, exist_ok=True)

    tree_df = pd.read_csv(treePath)

    print("Reading {} tree...".format(args.dataset))
    treeDic = {}
    for idx, row in tree_df.iterrows():
        eid, indexP, indexC = str(row["source_id"]), row["parent_idx"], row["self_idx"]
        max_degree, maxL, Vec = int(row["num_parent"]), int(row["max_seq_len"]), row["text"]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {"parent": indexP, "max_degree": max_degree, "maxL": maxL, "vec": Vec}
    print("tree no", len(treeDic))

    labelPath = "{}/{}_5fold/data.label.txt".format(args.data_root, args.dataset)
    labelset_nonR, labelset_f, labelset_t, labelset_u = ["non-rumor", "non-rumour"], ["false"], ["true"], ["unverified"]

    print("Loading tree label...")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0  ## T -> F -> U -> N
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid, label = str(line.split('\t')[0]), line.split('\t')[1]
        label = label.lower()
        event.append(eid)
        if label in labelset_t:
            labelDic[eid] = 0
            l1 += 1
        if label in labelset_f:
            labelDic[eid] = 1
            l2 += 1
        if label in labelset_u:
            labelDic[eid] = 2
            l3 += 1
        if label in labelset_nonR:
            labelDic[eid] = 3
            l4 += 1
    print(len(labelDic))
    print("T: {}, F: {}, U: {}, N: {}".format(l1, l2, l3, l4))

    def loadEid(event, id, y):
        if event is None:
            return None
        #if len(event) < 2:
        #	return None
        #if len(event)>1:
        #x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
        #x_x = getfeature(x_word, x_index)
        #rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(rootindex), np.array(y)

        ## Construct matrix with text content
        x_text, tree, root_text, rootindex = constructMat_txt(event)
        tree, rootindex, y = np.array(tree), np.array(rootindex), np.array(y)

        ## Features to be added
        ## - edgematrix (v)
        ## - root (v)
        ## - y (v)
        ## - rootindex (v)
        ## - nodecontent (v) - should contain content of responses only, not including root
        ## - topindex
        ## - triIndex

        #np.savez("{}/{}.npz".format(savePath, id), x=x_x, root=rootfeat, edgeindex=tree, rootindex=rootindex, y=y)
        np.savez("{}/{}.npz".format(savePath, id), nodecontent=x_text[1:], root=[root_text], edgematrix=tree,
                 rootindex=rootindex, y=y)
        return None

    print("Loading dataset...")
    Parallel(n_jobs=30, backend="threading")(
        delayed(loadEid)(treeDic[eid] if eid in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event))
    #Parallel(n_jobs=1, backend="threading")(delayed(loadEid)(treeDic[eid] if eid in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event))

    return


def split_5_fold(args):
    print("Splitting 5 fold for {}".format(args.dataset))

    path_i = "{}/{}".format(args.data_source, args.dataset.lower())
    path_o = "{}/{}_5fold".format(args.data_root, args.dataset)

    for fold in args.fold.split(","):
        print("Fold [{}]".format(fold))

        path_i_fold = "{}/split_{}".format(path_i, fold)
        path_o_fold = "{}/fold{}".format(path_o, fold)
        os.makedirs(path_o_fold, exist_ok=True)

        ## Read from RumorV2
        train_df = pd.read_csv("{}/train.csv".format(path_i_fold))
        test_df = pd.read_csv("{}/test.csv".format(path_i_fold))

        train_ids = train_df["source_id"].tolist()
        test_ids = test_df["source_id"].tolist()

        ## Write to DUCK
        with open("{}/_x_train.pkl".format(path_o_fold), "wb") as f:
            pickle.dump(train_ids, f)
        with open("{}/_x_test.pkl".format(path_o_fold), "wb") as f:
            pickle.dump(test_ids, f)


def make_label(args):
    print("Make `data.label.txt` for {}".format(args.dataset))

    path_i = "{}/{}/data.csv".format(args.data_source, args.dataset.lower())
    path_o = "{}/{}_5fold/data.label.txt".format(args.data_root, args.dataset)

    data_df = pd.read_csv(path_i)
    if not os.path.exists(f"{args.data_root}/{args.dataset}"):
        os.makedirs(f"{args.data_root}/{args.dataset}_5fold")

    with open(path_o, "w") as fw:
        eid_group = data_df.groupby("source_id")
        for eid, group in eid_group:
            fw.write("{}\t{}\n".format(eid, group["veracity"].tolist()[0]))


if __name__ == "__main__":
    args = parse_args()

    if args.split_5_fold:
        split_5_fold(args)
    elif args.build_graph:
        build_graph(args)
    elif args.make_label:
        make_label(args)
