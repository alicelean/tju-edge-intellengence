#!/bin/bash

echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
    /Users/alice1/opt/anaconda3/envs/tensorflow/bin/python  /Users/alice1/Desktop/tju.edu.com/python/adaptive-federated-learning-master/client.py&
     let "int++"
done