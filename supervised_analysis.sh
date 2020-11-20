#!/bin/bash


sudo CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python supervised_analyses.py ./data/supervised/discovery_train_x.csv ./data/supervised/discovery_train_y.csv ./results/save/supervised/discovery/ ./results/save/unsupervised/discovery/linear.txt ./results/save/unsupervised/discovery/model/linearautoencoder_0.h5

sudo CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python supervised_analyses.py ./data/supervised/replication_train_x.csv ./data/supervised/replication_train_y.csv ./results/save/supervised/replication/ ./results/save/unsupervised/replication/linear.txt ./results/save/unsupervised/replication/model/linearautoencoder_0.h5
