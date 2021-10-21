#!/bin/bash

echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
    /Users/alice/opt/anaconda3/envs/tensorflow/bin/python  /Users/alice/tju.com/python/tju-edge-intellengence/client.py&
     let "int++"
done