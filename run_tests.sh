# Take the first argument as the executable name
# Take the second argument as the sleep time
# Take the third argument as the number of runs

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_tests.sh <executable> <sleep_time> <num_runs>"
    exit 1
fi

# Check if the sleep time is a number
if [ ! -f "$1" ]; then
    echo "Executable $1 does not exist"
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
unamestr=$(uname -a)
start_time=$(date +%Y-%m-%d-%T)
output_log="${start_time}_log.outputs"
output_csv="${start_time}_outputs.csv"
output_folder="${start_time}_output_bin.outputs"
echo "Running tests $3 times, with $2 sec intervals, at $unamestr"
echo "Running tests $3 times, with $2 sec intervals, at $unamestr" > $output_log
echo "    Number of DSEs: ${num_dses[@]}"
echo "    Number of DSEs: ${num_dses[@]}" >> $output_log
echo "    DNNs: ${dnn_list[@]}"
echo "    DNNs: ${dnn_list[@]}" >> $output_log
echo "    Batch sizes: ${batch_list[@]}"
echo "    Batch sizes: ${batch_list[@]}" >> $output_log
echo "    Number of tiles: ${num_tiles[@]}"
echo "    Number of tiles: ${num_tiles[@]}" >> $output_log
echo "Script started at $start_time"
echo "Script started at $start_time" >> $output_log
echo ""
echo "" >> $output_log
mkdir $output_folder
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
                echo "    $(date +%T): Running $dnn with $num_dse DSEs, batch size $batch, and $num_tile tiles ($current_run/$total_runs)"
                echo "    $(date +%T): Running $dnn with $num_dse DSEs, batch size $batch, and $num_tile tiles ($current_run/$total_runs)" >> $output_log
                nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                output_format="cnn"
                #If dnn is bert_base, change output_format to bert
                if [ "$dnn" == "bert_base" ]; then
                    output_format="transformer"
                fi
                cmd="./$1 --sock_type=2 --pipelined=1 --dirname=temp --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --prefix="$dnn" --rx_ip="127.0.0.1" --rx_port=3786 --schedule_policy="local" --sched_sequential_idx=1 --dse_num=$num_dse --output_order="${output_format}" --inference_repeat_num=$3"
                echo "    $cmd" >> $output_log
                run_out=$(eval $cmd 2>&1)
                time_taken=$(echo $run_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                echo "        $run_out" >> $output_log
                echo "$dnn, $num_dse, $batch, $num_tile, ${time_taken} " >> $output_csv
                echo "" >> $output_log
                # Move the output file to the output folder if it exists
                if [ -f "aspen_output.bin" ]; then
                    mv aspen_output.bin $output_folder/${num_dse}_${dnn}_B${batch}_T${num_tile}.bin
                fi
                sleep $2
            done
        done
    done
done

echo "Script ended at $(date +%Y-%m-%d-%T)"
echo "Script ended at $(date +%Y-%m-%d-%T)" >> $output_log
