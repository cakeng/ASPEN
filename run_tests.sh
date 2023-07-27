# Take the first argument as the executable name
# Take the second argument as the sleep time
# Take the third argument as the number of runs
# Take the fourth argument as the number of DSEs on rx
# Take the fifth argument as the rx ip
# Take the sixth argument as the rx port

if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_tests.sh <executable> <sleep_time> <num_runs> <num_dse_rx> <rx_ip> <rx_port>"
    exit 1
fi

# Check if the sleep time is a number
if [ ! -f "$1" ]; then
    echo "Executable $1 does not exist"
    exit 1
fi
tx_dses=()
# Append the number of DSEs for each tx credential from a file named tx_dse_list.txt
if [ -f "tx_dse_list.txt" ]; then
    while read -r line; do
        tx_dses+=("$line")
    done < "tx_dse_list.txt"
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
policy_list=()
# Append the scheduling policies from a file named policy_list.txt
if [ -f "policy_list.txt" ]; then
    while read -r line; do
        policy_list+=("$line")
    done < "policy_list.txt"
fi
tx_list=()
# Append the tx credentials from a file named tx_cred_list.txt
if [ -f "tx_cred_list.txt" ]; then
    while read -r line; do
        tx_list+=("$line")
    done < "tx_cred_list.txt"
fi

total_runs=0
current_run=0
tx_cred_idx=0
rx_dse=$4
rx_ip=$5
rx_port=$6
rx_name=$(uname -n)
unamestr=$(uname -a)
start_time=$(date +%Y-%m-%d-%T)
output_log="${start_time}_log.outputs"
output_csv="${start_time}_outputs.csv"
output_folder="${start_time}_output_bin.outputs"
echo "Running tests $3 times, with $2 sec intervals, at $unamestr"
echo "Running tests $3 times, with $2 sec intervals, at $unamestr" > $output_log
echo "    DNNs: ${dnn_list[@]}"
echo "    DNNs: ${dnn_list[@]}" >> $output_log
echo "    Batch sizes: ${batch_list[@]}"
echo "    Batch sizes: ${batch_list[@]}" >> $output_log
echo "    Number of tiles: ${num_tiles[@]}"
echo "    Number of tiles: ${num_tiles[@]}" >> $output_log
echo "    Scheduling policies: ${policy_list[@]}"
echo "    Scheduling policies: ${policy_list[@]}" >> $output_log
echo "    Tx credentials: ${tx_list[@]}"
echo "    Tx credentials: ${tx_list[@]}" >> $output_log
echo "    Tx DSEs: ${tx_dses[@]}"
echo "    Tx DSEs: ${tx_dses[@]}" >> $output_log
echo ""
echo "Script started at $start_time"
echo "Script started at $start_time" >> $output_log
echo ""
echo "" >> $output_log
mkdir $output_folder
echo "TX_Cred, Scheduling Policy, DNN, Batch size, Number of tiles, Tx DSEs, Rx DSEs, Rx Time, Tx Time" >> $output_csv
for tx_cred in "${tx_list[@]}";
do
    for sched_policy in "${policy_list[@]}";
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
done

for tx_cred in "${tx_list[@]}";
do
    for sched_policy in "${policy_list[@]}";
    do
        for dnn in "${dnn_list[@]}";
        do
            for batch in "${batch_list[@]}";
            do
                for num_tile in "${num_tiles[@]}";
                do
                    current_run=$((current_run+1))
                    echo "    $(date +%T): Running $dnn with $rx_dse DSEs, batch size $batch, and $num_tile tiles, on tx $tx_cred ($current_run/$total_runs)"
                    echo "    $(date +%T): Running $dnn with $rx_dse DSEs, batch size $batch, and $num_tile tiles, on tx $tx_cred ($current_run/$total_runs)" >> $output_log
                    nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                    output_format="cnn"
                    #If dnn is bert_base, change output_format to bert
                    if [ "$dnn" == "bert_base" ]; then
                        output_format="transformer"
                    fi

                    tx_dse=${tx_dses[$tx_cred_idx]}
                    tx_name=$(ssh $tx_cred 'uname -n')
                    dir_name="Test_Rx_${rx_name}_Tx_${tx_name}"
                    rx_cmd="./$1 --sock_type=0 --pipelined=1 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --rx_ip="$rx_ip" --rx_port="$rx_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$rx_dse --output_order="${output_format}" --inference_repeat_num=$3"
                    tx_cmd_wo_ssh="./$1 --sock_type=1 --pipelined=1 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --rx_ip="$rx_ip" --rx_port="$rx_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$tx_dse --output_order="${output_format}" --inference_repeat_num=$3"
                    tx_cmd="ssh $tx_cred 'cd /home/cakeng/aspen_tests/ && $tx_cmd_wo_ssh'"
                    
                    echo "//////////    Rx command    //////////" >> $output_log
                    echo "    $rx_cmd" >> $output_log
                    echo "//////////    Tx command    //////////" >> $output_log
                    echo "    $tx_cmd" >> $output_log

                    #Run rx in background and store output in a temporary file
                    eval $rx_cmd 2>&1 | tee temp_rx_out.txt &
                    rx_pid=$!
                    sleep 1
                    #Run tx in foreground and store output in a temporary file
                    eval $tx_cmd 2>&1 | tee temp_tx_out.txt
                    tx_pid=$!
                    #Wait for rx to finish
                    wait $rx_pid
                    #Kill tx
                    wait $tx_pid
                    #Get the output from the temporary file
                    rx_out=$(cat temp_rx_out.txt)
                    tx_out=$(cat temp_tx_out.txt)
                    rm temp_rx_out.txt
                    rm temp_tx_out.txt
                    #Get the time taken from the output
                    rx_time_taken=$(echo $rx_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                    tx_time_taken=$(echo $tx_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                    #Print the output
                    echo "//////////    Rx Output    //////////" >> $output_log
                    echo "        $rx_out" >> $output_log
                    echo "//////////    Tx Output    //////////" >> $output_log
                    echo "        $tx_out" >> $output_log
                    echo "" >> $output_log
                    echo "${tx_cred}, ${sched_policy}, ${dnn}, ${batch}, ${num_tile}, ${tx_dses}, ${rx_dses}, ${rx_time_taken}, ${tx_time_taken}" >> $output_csv
                    sleep $2
                done
            done
        done
    done
    tx_cred_idx=$((tx_cred_idx+1))
done

echo "Script ended at $(date +%Y-%m-%d-%T)"
echo "Script ended at $(date +%Y-%m-%d-%T)" >> $output_log
