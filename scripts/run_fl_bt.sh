if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_fl.sh <device_mode>"
    exit 1
fi

DNN=vgg16
BATCH_SIZE=1
NUM_TILES=20
NUM_ITER=5
NUM_CORES_SERVER=16
NUM_CORES_EDGE=4

if [ ${1} -eq 0 ] ; then
    NUM_CORES=${NUM_CORES_SERVER}
elif [ ${1} -eq 1 ] ; then
    NUM_CORES=${NUM_CORES_EDGE}
fi

for fl_last_layer in {1..6}
do
    FL_LAST_LAYER=${fl_last_layer}

    for fl_num_path in {1..10}
    do
        FL_NUM_PATH=${fl_num_path}
        FL_PATH_OFFLOAD_IDX="0 1 0 1"

        ./main_fl \
            $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
            $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

    done
done





