# Take the first argument as the executable name
# Take the second argument as the sleep time
# Take the third argument as the number of runs
# Take the fourth argument as the SERVER ip
# Take the fifth argument as the SERVER port

if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_tests.sh <executable> <sleep_time> <num_runs> <server_ip> <server_port>"
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
edge_list=()
# Append the EDGE credentials from a file named edge_cred_list.txt
if [ -f "edge_cred_list.txt" ]; then
    while read -r line; do
        echo $line
        edge_list+=("$line")
    done < "edge_cred_list.txt"
fi
bandwidth_list=()
# Append the bandwidths from a file named bandwidth_list.txt
if [ -f "bandwidth_list.txt" ]; then
    while read -r line; do
        echo $line
        bandwidth_list+=("$line")
    done < "bandwidth_list.txt"
fi

total_runs=0
current_run=0
edge_cred_idx=0
server_ip=$4
server_port=$5
server_name=$(uname -n)
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
echo "    EDGE credentials: ${edge_list[@]}"
echo "    EDGE credentials: ${edge_list[@]}" >> $output_log
echo "    EDGE DSEs: ${edge_dse_nums[@]}"
echo "    EDGE DSEs: ${edge_dse_nums[@]}" >> $output_log
echo "    Network Bandwidth: ${bandwidth_list[@]}"
echo "    Network Bandwidth: ${bandwidth_list[@]}" >> $output_log
echo ""
echo "Script started at $start_time"
echo "Script started at $start_time" >> $output_log
echo ""
echo "" >> $output_log
mkdir $output_folder
echo "EDGE_Cred,Scheduling Policy,DNN,Batch size,Number of tiles,EDGE DSEs,SERVER DSEs,BWs,SERVER Time,EDGE Time,Total received ninsts" >> $output_csv
for edge_cred in "${edge_list[@]}";
do
    for sched_policy in "${policy_list[@]}";
    do
        for dnn in "${dnn_list[@]}";
        do
            for batch in "${batch_list[@]}";
            do
                for server_dse_num in "${server_dse_nums[@]}";
                do
                    for num_tile in "${num_tiles[@]}";
                    do
                        for bandwidth in "${bandwidth_list[@]}";
                        do
                            total_runs=$((total_runs+1))
                        done
                    done
                done
            done
        done
    done
done

for edge_cred in "${edge_list[@]}";
do
    for dnn in "${dnn_list[@]}";
    do
        for batch in "${batch_list[@]}";
        do
            for num_tile in "${num_tiles[@]}";
            do
                for server_dse_num in "${server_dse_nums[@]}";
                do
                    for bandwidth in "${bandwidth_list[@]}";
                    do
                        for sched_policy in "${policy_list[@]}";
                        do
                            current_run=$((current_run+1))
                            # server_dse_num=${server_dse_nums[$edge_cred_idx]}
                            edge_dse_num=${edge_dse_nums[$edge_cred_idx]}
                            echo "    $(date +%T): Running $dnn with $server_dse_num DSEs on server, $edge_dse_num DSEs on edge, batch size $batch, and $num_tile tiles, with EDGE $edge_cred BW ${bandwidth} ($current_run/$total_runs)"
                            echo "    $(date +%T): Running $dnn with $server_dse_num DSEs on server, $edge_dse_num DSEs on edge, batch size $batch, and $num_tile tiles, with EDGE $edge_cred BW ${bandwidth} ($current_run/$total_runs)" >> $output_log
                            nasm_file="${dnn}_B${batch}_T${num_tile}.nasm"
                            output_format="cnn"
                            #If dnn is bert_base, change output_format to bert
                            if [ "$dnn" == "bert_base" ]; then
                                output_format="transformer"
                            fi

                            password="\""bestnxcl\"""
                            shell_cmd="ssh $edge_cred"
                            # shell_cmd="adb -s $edge_cred shell"
                            edge_name=$(echo $edge_cred | cut -d' ' -f1)
                            echo "//////////    Set TC in EDGE    //////////" >> $output_log
                            tc_reset_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc qdisc del dev wlan0 root"
                            tc_reset_cmd="$shell_cmd '$tc_reset_cmd_wo_ssh'"
                            echo "     $tc_reset_cmd" >> $output_log
                            tc_set_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc qdisc add dev wlan0 root handle 1: htb default 6"
                            tc_set_cmd="$shell_cmd '$tc_set_cmd_wo_ssh'"
                            echo "     $tc_set_cmd" >> $output_log
                            tc_set_bw_cmd_wo_ssh="echo \"bestnxcl\" | sudo -S tc class add dev wlan0 parent 1: classid 1:6 htb rate ${bandwidth}mbit"
                            tc_set_bw_cmd="$shell_cmd '$tc_set_bw_cmd_wo_ssh'"
                            echo "     $tc_set_bw_cmd" >> $output_log
                            dir_name="Test_SERVER_${server_name}_EDGE_${edge_name}_${start_time}"
                            server_cmd="./$1 --device_mode=0 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$server_ip" --server_port="$server_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$server_dse_num --output_order="${output_format}" --inference_repeat_num=$3 --num_edge_devices=1"
                            edge_cmd_wo_ssh="./$1 --device_mode=1 --dirname=$dir_name --target_nasm_dir="data/$nasm_file" --target_dnn_dir="data/${dnn}_base.aspen" --target_input=data/batched_input_128.bin --prefix="$dnn" --server_ip="$server_ip" --server_port="$server_port" --schedule_policy="$sched_policy" --sched_sequential_idx=1 --dse_num=$edge_dse_num --output_order="${output_format}" --inference_repeat_num=$3 --num_edge_devices=1"
                            # edge_cmd="$shell_cmd 'cd /data/local/tmp/aspen_tests/ && $edge_cmd_wo_ssh'"
                            edge_cmd="$shell_cmd 'cd ~/kmbin/pipelining/aspen && $edge_cmd_wo_ssh'"
                            
                            echo "//////////    SERVER command    //////////" >> $output_log
                            echo "    $server_cmd" >> $output_log
                            echo "//////////    EDGE command    //////////" >> $output_log
                            echo "    $edge_cmd" >> $output_log

                            eval $tc_reset_cmd
                            eval $tc_set_cmd
                            eval $tc_set_bw_cmd
                            #Run SERVER in background and store output in a temporary file
                            eval $server_cmd 2>&1 | tee temp_server_out.tmp &
                            server_pid=$!
                            sleep 3
                            #Run EDGE in foreground and store output in a temporary file
                            eval $edge_cmd 2>&1 | tee temp_edge_out.tmp &
                            edge_pid=$!
                            #Wait max of 10 minutes for EDGE to finish
                            wait_time=600
                            while [ $wait_time -gt 0 ]; do
                                if ! kill -0 $edge_pid 2>/dev/null; then
                                    break
                                fi
                                sleep 1
                                wait_time=$((wait_time-1))
                            done
                            #If EDGE is still running, kill it
                            if kill -0 $edge_pid 2>/dev/null; then
                                echo "    $(date +%T): EDGE is still running after 600 seconds, killing it"
                                echo "    $(date +%T): EDGE is still running after 600 seconds, killing it" >> $output_log
                                kill -9 $edge_pid
                            fi
                            #Get the output from the temporary file
                            server_out=$(cat temp_server_out.tmp)
                            edge_out=$(cat temp_edge_out.tmp)
                            rm temp_server_out.tmp
                            rm temp_edge_out.tmp
                            
                            declare -A edge_total_received
                            declare -A edge_counts

                            while IFS= read -r line; do
                                if [[ $line =~ \[Edge\ ([0-9]+)\]\ Total\ received\ :\ \(([0-9]+)\/([0-9]+)\) ]]; then
                                    edge_num="${BASH_REMATCH[1]}"
                                    received="${BASH_REMATCH[2]}"
                                    total="${BASH_REMATCH[3]}"
                                    edge_total_received+="$received "
                                    ((edge_counts++))
                                fi
                            done <<< "$server_out"
                            #Get the time taken from the output
                            server_time_taken=$(echo $server_out | grep -oEi "Time measurement run_aspen \([0-9]+\): [0-9.]+ - ([0-9.]+) secs elapsed" | grep -oEi "([0-9.]+) secs elapsed" | grep -oE "[0-9.]+")
                            edge_time_taken=$(echo $edge_out | grep -oEi "Time measurement run_aspen \([0-9]+\): [0-9.]+ - ([0-9.]+) secs elapsed" | grep -oEi "([0-9.]+) secs elapsed" | grep -oE "[0-9.]+")
                            echo "${server_time_taken}" > time.temp
                            total_server_time=$(awk '{ sum += $1 } END { printf "%f", sum }' time.temp)
                            avg_server_time=$(echo "scale=6; $total_server_time/$3" | bc | awk '{printf "%f", $0}')
                            echo "${edge_time_taken}" > time.temp
                            total_edge_time=$(awk '{ sum += $1 } END { printf "%f", sum }' time.temp)
                            avg_edge_time=$(echo "scale=6; $total_edge_time/$3" | bc | awk '{printf "%f", $0}')
                            rm time.temp

                            total_received=0
                            total_count=${edge_counts}
                            received_values=(${edge_total_received})
                            
                            for received in "${received_values[@]}"; do
                                total_received=$((total_received + received))
                            done
                            
                            average_received=$((total_received / total_count))
                            echo "    $(date +%T): server took $avg_server_time seconds, edge took $avg_edge_time seconds with total received $average_received"
                            echo "    $(date +%T): server took $avg_server_time seconds, edge took $avg_edge_time seconds with total received $average_received" >> $output_log
                            #Print the output
                            echo "//////////    SERVER Output    //////////" >> $output_log
                            echo "        $server_out" >> $output_log
                            echo "//////////    EDGE Output    //////////" >> $output_log
                            echo "        $edge_out" >> $output_log
                            echo "" >> $output_log
                            echo "${edge_cred},${sched_policy},${dnn},${batch},${num_tile},${edge_dse_num},${server_dse_num},${bandwidth},${avg_server_time},${avg_edge_time},${average_received}" >> $output_csv
                            sleep $2
                        done
                    done
                done
            done
        done
    done
    edge_cred_idx=$((edge_cred_idx+1))
done

echo "Script ended at $(date +%Y-%m-%d-%T)"
echo "Script ended at $(date +%Y-%m-%d-%T)" >> $output_log
