#!/bin/bash

echo "开始测试......"
n_nodes=5
int=1
while(( $int<=$n_nodes ))
do
    /usr/local/bin/python3.7  /Users/alice1/Desktop/tju.edu.com/python/tju-edge-intellengence/client.py&
     let "int++"
done