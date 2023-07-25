# Take the first argument as the sleep time
# Take the second argument as the executable name
# Check if the sleep time is a number
if [ ! -f "$2" ]; then
    echo "Executable $2 does not exist"
    exit 1
fi
num_dses=()
# Append the number of DSEs from a file named num_dse_list.txt
if [ -f "num_dse_list.txt" ]; then
    while read -r line; do
        num_dses+=("$line")
    done < "num_dse_list.txt"
fi
dnn_list=()
# Append the DNNs from a file named dnn_list.txt
if [ -f "dnn_list.txt" ]; then
    while read -r line; do
        dnn_list+=("$line")
    done < "dnn_list.txt"
fi
batch_list=()
# Append the batch sizes from a file named batch_list.txt
if [ -f "batch_list.txt" ]; then
    while read -r line; do
        batch_list+=("$line")
    done < "batch_list.txt"
fi
num_tiles=()
# Append the number of tiles from a file named num_tiles_list.txt
if [ -f "num_tiles_list.txt" ]; then
    while read -r line; do
        num_tiles+=("$line")
    done < "num_tiles_list.txt"
fi
total_runs=0
current_run=0
output_log="output.log"
unamestr=$(uname -a)
echo "Running tests for $1 at $unamestr"
echo "Running tests for $1 at $unamestr" > $output_log
echo "Number of DSEs: ${num_dses[@]}"
echo "Number of DSEs: ${num_dses[@]}" >> $output_log
echo "DNNs: ${dnn_list[@]}"
echo "DNNs: ${dnn_list[@]}" >> $output_log
echo "Batch sizes: ${batch_list[@]}"
echo "Batch sizes: ${batch_list[@]}" >> $output_log
echo "Number of tiles: ${num_tiles[@]}"
echo "Number of tiles: ${num_tiles[@]}" >> $output_log
echo "Script started at $(date)"
echo "Script started at $(date)" >> $output_log
echo ""
echo "" >> $output_log
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
                nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                cmd="./$2 --sock_type=2 --pipelined=1 --dirname=temp --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --prefix="$dnn" --rx_ip="127.0.0.1" --rx_port=3786 --schedule_policy="local" --sched_sequential_idx=1 --dse_num=$num_dse --output_order="cnn" --inference_repeat_num=20"
                echo "    $cmd" >> $output_log
                run_out=$(eval $cmd)
                time_taken=$(echo $run_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                echo "        $run_out" >> $output_log
                echo "TIME_LOG, $num_dse, $dnn, $batch, $num_tile, ${time_taken} " >> $output_log
                echo "" >> $output_log
                sleep $1
            done
        done
    done
done
echo "Script ended at $(date)"
echo "Script ended at $(date)" >> $output_log
