#!/bin/bash


sudo CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python execute_autoencoder.py ./data/discovery_data/train_dump_sMRI_socialbrain_sym_r2.5_s5 ./results/save/unsupervised/discovery/

sudo CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python execute_autoencoder.py ./data/replication_data/train_dump_sMRI_socialbrain_sym_r2.5_s5 ./results/save/unsupervised/replication/
