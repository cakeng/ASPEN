source ./scripts/param_dynamic.sh

./main_scheduling \
    --sock_type=1   \
    --pipelined=${PIPELINED}  \
    --dirname=${DIRNAME}  \
    --target_dnn_dir=${TARGET_DNN_DIR} \
    --target_nasm_dir=${TARGET_NASM_DIR} \
    --target_input=${TARGET_INPUT} \
    --prefix=${PREFIX} \
    --rx_ip=${RX_IP} \
    --rx_port=${RX_PORT}   \
    --schedule_policy=${SCHEDULE_POLICY}  \
    --sched_sequential_idx=${SCHED_SEQUENTIAL_IDX}    \
    --dse_num=${DSE_NUM}    \
    --output_order=${OUTPUT_ORDER}    \
    --inference_repeat_num=${INFERENCE_REPEAT_NUM}