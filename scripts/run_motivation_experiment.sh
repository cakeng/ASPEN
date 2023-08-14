#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_conventional_experiment.sh <sever_ip> <sever_port> <num_runs>"
    exit 1
fi

start_time=$(date +%Y-%m-%d-%T)
echo "$start_time"

output_log="./logs/motivation/${start_time}_log.outputs"
output_csv="./logs/motivation/${start_time}_outputs.csv"

edge_cred="nxc@192.168.1.25"
server_name=$(uname -n)
edge_name=$(echo $edge_cred | cut -d' ' -f1)

dir_name="Test_SERVER_${server_name}_EDGE_${edge_name}_${start_time}"
server_ip=$1
server_port=$2
inference_repeat_num=$3
batch=1
num_tile=100

NUM_EDGE_DEVICES=1

nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"

schedule_policy='conventional'


dnn_list=("resnet50" "bert_base" "yolov3" "vgg16")
bw_list=(10 50 80)
server_dse_num_list=(1 4 8 16)
edge_dse_num_list=(4 8)
split_layer_list=()

echo "EDGE_Cred,Split Idx,DNN,Batch size,Number of tiles,EDGE DSEs,SERVER DSEs,BWs,Edge Computing Time (ms),Communication Time (ms),Server Computing Time (ms),Computing Time (ms)" >> $output_csv

for dnn in "${dnn_list[@]}"
do
    if [[ "$dnn" == *"vgg16"* ]]; then
        for ((i=1; i<=20; i++)); do
            split_layer_list+=("$i")
        done
    fi

    if [[ "$dnn" == *"resnet50"* ]]; then
        for ((i=1; i<=72; i++)); do
            split_layer_list+=("$i")
        done
    fi

    if [[ "$dnn" == *"bert_base"* ]]; then
        for ((i=1; i<=144; i++)); do
            split_layer_list+=("$i")
        done
    fi

    if [[ "$dnn" == *"yolov3"* ]]; then
        for ((i=1; i<=103; i++)); do
            split_layer_list+=("$i")
        done
    fi

    output_format="cnn"
    #If dnn is bert_base, change output_format to bert
    if [ "$dnn" == "bert_base" ]; then
        output_format="transformer"
    fi
    
    for bw in "${bw_list[@]}"
    do
        for server_dse_num in "${server_dse_num_list[@]}"
        do
            for edge_dse_num in "${edge_dse_num_list[@]}"
            do
                for split_idx in "${split_layer_list[@]}"
                do
                    eval "killall main_mu"
                    nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                    shell_cmd="ssh ${edge_cred}"
                    echo "//////////    Set TC in EDGE    //////////" >> $output_log
                    tc_reset_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc qdisc del dev wlan0 root"
                    tc_reset_cmd="$shell_cmd '$tc_reset_cmd_wo_ssh'"
                    echo "     $tc_reset_cmd" >> $output_log
                    tc_set_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc qdisc add dev wlan0 root handle 1: htb default 6"
                    tc_set_cmd="$shell_cmd '$tc_set_cmd_wo_ssh'"
                    echo "     $tc_set_cmd" >> $output_log
                    tc_set_bw_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc class add dev wlan0 parent 1: classid 1:6 htb rate ${bw}mbit"
                    tc_set_bw_cmd="$shell_cmd '$tc_set_bw_cmd_wo_ssh'"
                    echo "     $tc_set_bw_cmd" >> $output_log

                    eval $tc_reset_cmd
                    eval $tc_set_cmd
                    eval $tc_set_bw_cmd
                    eval "$shell_cmd 'killall main_mu'"

                    server_cmd="./main_mu --device_mode=0 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$server_ip" --server_port="$server_port" --schedule_policy="$schedule_policy" --sched_sequential_idx=$split_idx --dse_num=$server_dse_num --output_order="${output_format}" --inference_repeat_num=$inference_repeat_num --num_edge_devices=$NUM_EDGE_DEVICES"
                    edge_cmd_wo_ssh="./main_mu --device_mode=1 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$server_ip" --server_port="$server_port" --schedule_policy="$schedule_policy" --sched_sequential_idx=$split_idx --dse_num=$edge_dse_num --output_order="${output_format}" --inference_repeat_num=$inference_repeat_num --num_edge_devices=$NUM_EDGE_DEVICES"

                    
                    edge_cmd="$shell_cmd 'cd ./kmbin/pipelining/aspen && $edge_cmd_wo_ssh'"

                    echo "//////////    SERVER command    //////////" >> $output_log
                    echo "    $server_cmd" >> $output_log
                    echo "$server_cmd"
                    echo "//////////    EDGE command    //////////" >> $output_log
                    echo "    $edge_cmd" >> $output_log
                    echo "$edge_cmd"

                    eval $server_cmd 2>&1 | tee temp_server_out.tmp &
                    server_pid=$!
                    sleep 2
                    
                    #Run EDGE in foreground and store output in a temporary file
                    edge_pid_list=()
                    edge_out_list=()
                    for i in $(seq 1 $NUM_EDGE_DEVICES)
                    do
                        echo "    $(date +%T): edge command: ${edge_cmd}"
                        eval $edge_cmd 2>&1 | tee temp_edge_out${i}.tmp &
                        edge_pid_list+=($!)
                        echo "    $(date +%T): edge PID: ${edge_pid_list[$i-1]}"
                    done

                    wait_time=600
                    total_finish=0
                    while [ $wait_time -gt 0 ]; do
                        for edge_pid in "${edge_pid_list[@]}"
                        do
                            if ! kill -0 $edge_pid 2>/dev/null; then
                                total_finish=$((total_finish+1))
                            fi
                        done
                        if [ $total_finish -ge $NUM_EDGE_DEVICES ]; then
                            break
                        fi
                        sleep 1
                        wait_time=$((wait_time-1))
                    done
                    #If EDGE is still running, kill it
                    for edge_pid in "${edge_pid_list[@]}"
                    do
                        if kill -0 $edge_pid 2>/dev/null; then
                            echo "    $(date +%T): EDGE is still running after 600 seconds, killing it"
                            kill -9 $edge_pid
                        fi
                    done

                    server_out=$(cat temp_server_out.tmp)
                    echo "        $server_out" >> $output_log
                    rm -f temp_server_out.tmp

                    for i in $(seq 1 $NUM_EDGE_DEVICES)
                    do
                        edge_out=$(cat temp_edge_out${i}.tmp)
                        rm -f temp_edge_out${i}.tmp
                        echo "        Edge ${i} output:" >> $output_log
                        echo "        $edge_out" >> $output_log
                    done 

                    for ((edge_id=0; edge_id<NUM_EDGE_DEVICES; edge_id++));
                    do
                        edge_compute_diff_sum=0
                        server_compute_diff_sum=0
                        communicate_diff_sum=0

                        for ((iter_num=0; iter_num<inference_repeat_num; iter_num++)); 
                        do
                            remote_path="kmbin/pipelining/aspen/logs/${dir_name}/edge_${edge_id}/"
                            local_path="logs/motivation/${dir_name}/edge_${edge_id}/"
                            local_log_path="logs/${dir_name}/edge_${edge_id}/"
                            mkdir -p $local_path
                            remote_filename="${dnn}_${schedule_policy}_EDGE_${dnn}_B${batch}_T${num_tile}_Iter${iter_num}.csv"
                            local_log_filename="${dnn}_${schedule_policy}_SERVER_${dnn}_B${batch}_T${num_tile}_Iter${iter_num}.csv"
                            local_edge_filename="${dnn}_${schedule_policy}_EDGE_${dnn}_B${batch}_T${num_tile}_Split${split_idx}_Iter${iter_num}.csv"
                            local_server_filename="${dnn}_${schedule_policy}_SERVER_${dnn}_B${batch}_T${num_tile}_Split${split_idx}_Iter${iter_num}.csv"
                            
                            cp -f ${local_log_path}${local_log_filename} ${local_path}${local_server_filename}

                            sftp "$edge_cred:${remote_path}${remote_filename}" "${local_path}${local_edge_filename}"

                            # Edge Compute time
                            edge_max_compute_time=($(tail -n +2 ${local_path}${local_edge_filename} | cut -d ',' -f3 | grep -v "0.000000" | sort -n | tail -n 1))
                            edge_min_compute_time=($(tail -n +2 ${local_path}${local_edge_filename} | cut -d ',' -f3 | grep -v "0.000000" | sort -r | tail -n 1))
                            if [ -z "$edge_max_compute_time" ]; then
                                edge_max_compute_time=0
                            fi

                            if [ -z "$edge_min_compute_time" ]; then
                                edge_min_compute_time=0
                            fi

                            edge_compute_diff=$(bc -l <<< "$edge_max_compute_time - $edge_min_compute_time")
                            edge_compute_diff_sum=$(bc -l <<< "$edge_compute_diff_sum + $edge_compute_diff")

                            # Server Compute time
                            max_compute_time=($(tail -n +2 ${local_path}${local_server_filename} | cut -d ',' -f3 | grep -v "0.000000" | sort -n | tail -n 1))
                            min_compute_time=($(tail -n +2 ${local_path}${local_server_filename} | cut -d ',' -f3 | grep -v "0.000000" | sort -r | tail -n 1))
                            server_compute_diff=$(bc -l <<< "$max_compute_time - $min_compute_time")
                            server_compute_diff_sum=$(bc -l <<< "$server_compute_diff_sum + $server_compute_diff")
                            
                            # Communication time
                            min_sent_time=($(tail -n +2 ${local_path}${local_edge_filename} | cut -d ',' -f5 | grep -v "0.000000" | sort -r | tail -n 1))  
                            max_recv_time=($(tail -n +2 ${local_path}${local_server_filename} | cut -d ',' -f4 | grep -v "0.000000" | sort -n | tail -n 1 ))
                            communicate_diff=$(echo "$max_recv_time - $min_sent_time" | bc -l)
                            communicate_diff_sum=$(echo "$communicate_diff_sum + $communicate_diff" | bc -l)
                        done

                        avg_edge_compute=$(bc -l <<< "$edge_compute_diff_sum / $inference_repeat_num")
                        avg_server_compute=$(bc -l <<< "$server_compute_diff_sum / $inference_repeat_num")
                        avg_communicate=$(bc -l <<< "$communicate_diff_sum / $inference_repeat_num")
                        avg_compute=$(bc -l <<< "$avg_edge_compute + $avg_server_compute")

                        echo "${edge_cred},${split_idx},${dnn},${batch},${num_tile},${edge_dse_num},${server_dse_num},${bw},${avg_edge_compute},${avg_communicate},${avg_server_compute},${avg_compute}" >> $output_csv
                        
                        echo "Edge ${edge_id} Split Idx: ${split_idx} - Average Edge Compute Time (ms): ${avg_edge_compute}" >> $output_log
                        echo "Edge ${edge_id} Split Idx: ${split_idx} - Average Server Compute Time (ms): ${avg_server_compute}" >> $output_log
                        echo "Edge ${edge_id} Split Idx: ${split_idx} - Average Edge+Server Compute Time (ms): ${avg_compute}" >> $output_log
                        echo "Edge ${edge_id} Split Idx: ${split_idx} - Average Communication Time (ms): ${avg_communicate}" >> $output_log
                    done
                done
            done
        done
    done
done