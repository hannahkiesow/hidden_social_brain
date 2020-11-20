#!/bin/bash


sudo CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0  python train_test_split.py

sudo CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0  python supervised_data_preparation.py
