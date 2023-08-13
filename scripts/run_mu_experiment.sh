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

edge_exec_cmd="./main_scheduling \
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
    --dse_num=${DSE_NUM}    \
    --output_order=${OUTPUT_ORDER}    \
    --inference_repeat_num=${INFERENCE_REPEAT_NUM}"

edge_cmd="${shell_cmd} \"${edge_exec_cmd}\""

echo $server_cmd
eval $server_cmd
echo $edge_cmd
for i in $(seq 1 $NUM_EDGE_DEVICES)
do
    eval $edge_cmd
done