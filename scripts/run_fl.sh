if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_fl.sh <device_mode>"
    exit 1
fi

source ./scripts/param_fl.sh

if [ ${1} -eq 0 ] ; then
    NUM_CORES=${NUM_CORES_SERVER}
elif [ ${1} -eq 1 ] ; then
    NUM_CORES=${NUM_CORES_EDGE}
fi

./main_fl \
    $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
    $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

