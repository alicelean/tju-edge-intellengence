#!/bin/bash

echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
    /home/yinqiaoyun/anaconda3/envs/tensorflow/bin/python  ~/programe/python/tju-edge-intellengence/client.py&
     let "int++"
done