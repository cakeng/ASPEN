#!/bin/bash

if [ "$#" -ne 9 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./download_logs.sh <host> <ip> <port> <model> <batch> <num_tiles> <dir_name> <edge_id> <num_runs>"
    exit 1
fi

host=$1
ip=$2
port=$3
model=$4
batch=$5
num_tiles=$6
dir_name=$7
edge_id=$8
num_runs=$9

echo "--------------------------------------------------------------------------------"
echo "Downloading logs from Edge $edge_id $host@$ip:$port"
echo "Model: $model, Batch: $batch, Num Tiles: $num_tiles, Dir Name: $dir_name, Num Runs: $num_runs"
echo "--------------------------------------------------------------------------------"

path="kmbin/pipelining/aspen/logs/$dir_name/edge_$edge_id"
target_path="logs/$dir_name/edge_$edge_id"


for ((i=0; i<$num_runs; i++))
do
    conventional_file_name="${model}_conventional_EDGE_${model}_B${batch}_T${num_tiles}_Iter$i.csv"
    dynamic_file_name="${model}_dynamic_EDGE_${model}_B${batch}_T${num_tiles}_Iter$i.csv"

    cmd="scp -P $port $host@$ip:$path/$conventional_file_name $target_path/$conventional_file_name"
    eval $cmd
    cmd="scp -P $port $host@$ip:$path/$dynamic_file_name $target_path/$dynamic_file_name"
    eval $cmd
done




