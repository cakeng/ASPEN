# Take the first argument as the executable name
# Take the second argument as the sleep time
# Take the third argument as the number of runs
# Take the fourth argument as the SERVER ip
# Take the fifth argument as the SERVER port

if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_tests.sh <executable> <sleep_time> <num_runs> <num_dse_SERVER> <SERVER_ip> <SERVER_port>"
    exit 1
fi

# Check if the sleep time is a number
if [ ! -f "$1" ]; then
    echo "Executable $1 does not exist"
    exit 1
fi
edge_dse_nums=()
# Append the number of DSEs for each EDGE credential from a file named edge_dse_num_list.txt
if [ -f "edge_dse_num_list.txt" ]; then
    while read -r line; do
        edge_dse_nums+=("$line")
    done < "edge_dse_num_list.txt"
fi
server_dse_nums=()
# Append the number of DSEs for each SERVER credential from a file named server_dse_num_list.txt
if [ -f "server_dse_num_list.txt" ]; then
    while read -r line; do
        server_dse_nums+=("$line")
    done < "server_dse_num_list.txt"
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
EDGE_list=()
# Append the EDGE credentials from a file named edge_cred_list.txt
if [ -f "edge_cred_list.txt" ]; then
    while read -r line; do
        EDGE_list+=("$line")
    done < "edge_cred_list.txt"
fi

total_runs=0
current_run=0
edge_cred_idx=0
SERVER_ip=$4
SERVER_port=$5
SERVER_name=$(uname -n)
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
echo "    EDGE credentials: ${EDGE_list[@]}"
echo "    EDGE credentials: ${EDGE_list[@]}" >> $output_log
echo "    EDGE DSEs: ${edge_dse_nums[@]}"
echo "    EDGE DSEs: ${edge_dse_nums[@]}" >> $output_log
echo ""
echo "Script started at $start_time"
echo "Script started at $start_time" >> $output_log
echo ""
echo "" >> $output_log
mkdir $output_folder
echo "EDGE_Cred, Scheduling Policy, DNN, Batch size, Number of tiles, EDGE DSEs, SERVER DSEs, SERVER Time, EDGE Time" >> $output_csv
for edge_cred in "${EDGE_list[@]}";
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

for edge_cred in "${EDGE_list[@]}";
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
                    echo "    $(date +%T): Running $dnn with $server_dse_num DSEs, batch size $batch, and $num_tile tiles, on EDGE $edge_cred ($current_run/$total_runs)"
                    echo "    $(date +%T): Running $dnn with $server_dse_num DSEs, batch size $batch, and $num_tile tiles, on EDGE $edge_cred ($current_run/$total_runs)" >> $output_log
                    nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                    output_format="cnn"
                    #If dnn is bert_base, change output_format to bert
                    if [ "$dnn" == "bert_base" ]; then
                        output_format="transformer"
                    fi
                    server_dse_num=${server_dse_nums[$edge_cred_idx]}
                    edge_dse_num=${edge_dse_nums[$edge_cred_idx]}

                    EDGE_name=$(ssh $edge_cred 'uname -n')
                    dir_name="Test_SERVER_${SERVER_name}_EDGE_${EDGE_name}"
                    server_cmd="./$1 --sock_type=0 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$SERVER_ip" --server_port="$SERVER_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$server_dse_num --output_order="${output_format}" --inference_repeat_num=$3"
                    EDGE_cmd_wo_ssh="./$1 --sock_type=1 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$SERVER_ip" --server_port="$SERVER_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$edge_dse_num --output_order="${output_format}" --inference_repeat_num=$3"
                    EDGE_cmd="ssh $edge_cred 'cd /home/cakeng/aspen_tests/ && $EDGE_cmd_wo_ssh'"
                    
                    echo "//////////    SERVER command    //////////" >> $output_log
                    echo "    $server_cmd" >> $output_log
                    echo "//////////    EDGE command    //////////" >> $output_log
                    echo "    $EDGE_cmd" >> $output_log

                    #Run SERVER in background and store output in a temporary file
                    eval $server_cmd 2>&1 | tee temp_server_out.txt &
                    server_pid=$!
                    sleep 1
                    #Run EDGE in foreground and store output in a temporary file
                    eval $EDGE_cmd 2>&1 | tee temp_edge_out.txt
                    edge_pid=$!
                    #Wait for SERVER to finish
                    wait $server_pid
                    #Kill EDGE
                    wait $edge_pid
                    #Get the output from the temporary file
                    server_out=$(cat temp_server_out.txt)
                    edge_out=$(cat temp_edge_out.txt)
                    rm temp_server_out.txt
                    rm temp_edge_out.txt
                    #Get the time taken from the output
                    SERVER_time_taken=$(echo $server_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                    EDGE_time_taken=$(echo $edge_out | grep -oEi "Time taken: ([0-9.]+) seconds" | grep -oE "[0-9.]+")
                    #Print the output
                    echo "//////////    SERVER Output    //////////" >> $output_log
                    echo "        $server_out" >> $output_log
                    echo "//////////    EDGE Output    //////////" >> $output_log
                    echo "        $edge_out" >> $output_log
                    echo "" >> $output_log
                    echo "${edge_cred}, ${sched_policy}, ${dnn}, ${batch}, ${num_tile}, ${edge_dse_num}, ${server_dse_num}, ${SERVER_time_taken}, ${EDGE_time_taken}" >> $output_csv
                    sleep $2
                done
            done
        done
    done
    edge_cred_idx=$((edge_cred_idx+1))
done

echo "Script ended at $(date +%Y-%m-%d-%T)"
echo "Script ended at $(date +%Y-%m-%d-%T)" >> $output_log
