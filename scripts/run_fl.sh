if [ "$#" -gt 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_fl.sh <device_mode> [opt:gdb]"
    exit 1
fi

source ./scripts/param_fl.sh

if [ ${1} -eq 0 ] ; then
    NUM_CORES=${NUM_CORES_SERVER}
elif [ ${1} -gt 0 ] ; then
    NUM_CORES=${NUM_CORES_EDGE}
fi

if [ "$#" -gt 1 ] ; then
    gdb --args ./main_fl \
        $DNN $NUM_ITER $NUM_CORES $1
else
    ./main_fl \
        $DNN $NUM_ITER $NUM_CORES $1
fi



