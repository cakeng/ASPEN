if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_fl.sh <device_mode>"
    exit 1
fi

source ./scripts/param_fl.sh

./main_fl \
    $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
    $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

