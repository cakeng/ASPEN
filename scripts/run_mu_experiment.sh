source ./scripts/param_mu.sh

server_cmd="./main_mu \
    --device_mode=0   \
    --dirname=${DIRNAME}  \
    --target_dnn_dir=${TARGET_DNN_DIR} \
    --target_nasm_dir=${TARGET_NASM_DIR} \
    --target_input=${TARGET_INPUT} \
    --prefix=${PREFIX} \
    --server_ip=${server_ip} \
    --server_port=${server_port}   \
    --schedule_policy=${SCHEDULE_POLICY}  \
    --sched_sequential_idx=${SCHED_SEQUENTIAL_IDX}    \
    --dse_num=${DSE_NUM}    \
    --output_order=${OUTPUT_ORDER}    \
    --inference_repeat_num=${INFERENCE_REPEAT_NUM} \
    --num_edge_devices=${NUM_EDGE_DEVICES}"

shell_cmd="ssh ${EDGE_CREDIT}"

edge_exec_cmd="./main_mu \
    --device_mode=1   \
    --dirname=${DIRNAME}  \
    --target_dnn_dir=${TARGET_DNN_DIR} \
    --target_nasm_dir=${TARGET_NASM_DIR} \
    --target_input=${TARGET_INPUT} \
    --prefix=${PREFIX} \
    --server_ip=${server_ip} \
    --server_port=${server_port}   \
    --schedule_policy=${SCHEDULE_POLICY}  \
    --sched_sequential_idx=${SCHED_SEQUENTIAL_IDX}    \
    --dse_num=${EDGE_DSE_NUM}    \
    --output_order=${OUTPUT_ORDER}    \
    --inference_repeat_num=${INFERENCE_REPEAT_NUM} \
    --num_edge_devices=${NUM_EDGE_DEVICES}"

edge_cmd="$shell_cmd 'cd ./kmbin/pipelining/aspen && $edge_exec_cmd'"
echo "    $(date +%T): server command: ${server_cmd}"
eval $server_cmd 2>&1 | tee temp_server_out.tmp &
server_pid=$!
sleep 3
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

#Wait max of 10 minutes for EDGE to finish
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
#Get the output from the temporary file
server_out=$(cat temp_server_out.tmp)
# rm temp_server_out.tmp

echo "$(date +%T): scheduling_policy: ${SCHEDULE_POLICY}"
#Get the time taken from the output
server_time_taken=$(echo $server_out | grep -oEi "Time measurement run_aspen \([0-9]+\): [0-9.]+ - ([0-9.]+) secs elapsed" | grep -oEi "([0-9.]+) secs elapsed" | grep -oE "[0-9.]+")
echo "${server_time_taken}" > time.temp
total_server_time=$(awk '{ sum += $1 } END { printf "%f", sum }' time.temp)
avg_server_time=$(echo "scale=6; $total_server_time/$INFERENCE_REPEAT_NUM/$NUM_EDGE_DEVICES" | bc | awk '{printf "%f", $0}')
echo "$(date +%T): server took $avg_server_time seconds"

for i in $(seq 1 $NUM_EDGE_DEVICES)
do
    edge_out=$(cat temp_edge_out$i.tmp)
    # rm temp_edge_out$i.tmp
    edge_time_taken=$(echo $edge_out | grep -oEi "Time measurement run_aspen \([0-9]+\): [0-9.]+ - ([0-9.]+) secs elapsed" | grep -oEi "([0-9.]+) secs elapsed" | grep -oE "[0-9.]+")
    echo "${edge_time_taken}" > time.temp
    total_edge_time=$(awk '{ sum += $1 } END { printf "%f", sum }' time.temp)
    avg_edge_time=$(echo "scale=6; $total_edge_time/$INFERENCE_REPEAT_NUM" | bc | awk '{printf "%f", $0}')
    rm time.temp
    echo "$(date +%T): edge $i took $avg_edge_time seconds"
done

