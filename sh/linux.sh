#!/bin/bash

echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
     /home/alice/anaconda3/envs/tju/bin/python  /home/alice/python/tju-edge-intellengence/client.py&
     let "int++"
done