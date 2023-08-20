source ./scripts/param_dynamic.sh

cmd="./main_scheduling \
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
    --sched_partial_ratio=${SCHED_PARTIAL_RATIO}    \
    --dse_num=${DSE_NUM}    \
    --output_order=${OUTPUT_ORDER}    \
    --inference_repeat_num=${INFERENCE_REPEAT_NUM}"

echo $cmd
eval $cmd