#! /bin/bash

# Take the first argument as the executable name

num_dses=("1" "2" "4" "6" "8")
dnn_list=("resnet50" "vgg16" "yolov3" "bert_base")
batch_list=("1" "2" "4" "8" "16" "32")
num_tiles=("20" "50" "100" "200")
total_runs=0
current_run=0
output_log="output.log"
echo "Running tests for $1"
echo "Running tests for $1" > $output_log
echo "Script started at $(date)"
echo "Script started at $(date)" >> $output_log
for num_dse in "${num_dses[@]}"; 
do
    for dnn in "${dnn_list[@]}";
    do
        for batch in "${batch_list[@]}";
        do
            for num_tile in "${num_tiles[@]}";
            do
                total_runs=$((total_runs+1))
            done
        done
    done
done
for num_dse in "${num_dses[@]}"; 
do
    for dnn in "${dnn_list[@]}";
    do
        for batch in "${batch_list[@]}";
        do
            for num_tile in "${num_tiles[@]}";
            do
                current_run=$((current_run+1))
                echo "    Running $dnn with $num_dse DSEs, batch size $batch, and $num_tile tiles ($current_run/$total_runs)"
                echo "    Running $dnn with $num_dse DSEs, batch size $batch, and $num_tile tiles ($current_run/$total_runs)" >> $output_log
                nasm_file="$dnn_B$batch_T$num_tile.nasm"
                cmd="./$1 --sock_type=1 --pipelined=1 --dirname=temp \\ 
                --target_nasm_dir=\"data/vgg16_B1.nasm\" \\
                --target_dnn_dir=\"data/${dnn}_base.aspen\" \\
                --prefix=\"$dnn\" \\
                --rx_ip=\"127.0.01\" \\
                --rx_port=3786 \\
                --schedule_policy=\"local\" \\
                --sched_sequential_idx=1 \\
                --dse_num=$num_dse \\
                --output_order=\"cnn\" \\
                --inference_repeat_num=20\""
                echo "    $cmd"
                echo "    $cmd" >> $output_log
                $cmd >> $output_log
                echo ""
                echo "" >> $output_log
            done
        done
    done
done
echo "Script ended at $(date)"
echo "Script ended at $(date)" >> $output_log