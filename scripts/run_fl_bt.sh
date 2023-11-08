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
    ## SERVER ##
    NUM_CORES=${NUM_CORES_SERVER}
    SLEEP_BETWEEN_ITER=0

elif [ ${1} -eq 1 ] ; then
    ### EDGE ###
    NUM_CORES=${NUM_CORES_EDGE}
    SLEEP_BETWEEN_ITER=2
fi

for fl_last_layer in {1..6}
do
    FL_LAST_LAYER=${fl_last_layer}

    ### PATH 1 ###

    FL_NUM_PATH=1

    for ((i=0; i<${fl_last_layer}; i++))
    do
        FL_PATH_OFFLOAD_IDX=$i

        echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

        ./main_fl \
            $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
            $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

        sleep ${SLEEP_BETWEEN_ITER}
    done

    
    ### PATH 2 ###

    FL_NUM_PATH=2

    for ((i1=0; i1<${fl_last_layer}; i1++))
    do
        for ((i2=0; i2<=${fl_last_layer}; i2++))
        do
            FL_PATH_OFFLOAD_IDX="$i1 $i2"

            echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

            ./main_fl \
                $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
                $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

            sleep ${SLEEP_BETWEEN_ITER}
        done
    done

    ### PATH 3 ###

    FL_NUM_PATH=3

    for ((i1=0; i1<${fl_last_layer}; i1++))
    do
        for ((i2=0; i2<=${fl_last_layer}; i2++))
        do
            for ((i3=0; i3<=${fl_last_layer}; i3++))
            do
                FL_PATH_OFFLOAD_IDX="$i1 $i2 $i3"

                echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

                ./main_fl \
                    $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
                    $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

                sleep ${SLEEP_BETWEEN_ITER}
            done
        done
    done

    ### PATH 4 ###

    FL_NUM_PATH=4

    for ((i1=0; i1<${fl_last_layer}; i1++))
    do
        for ((i2=0; i2<=${fl_last_layer}; i2++))
        do
            for ((i3=0; i3<=${fl_last_layer}; i3++))
            do
                for ((i4=0; i4<=${fl_last_layer}; i4++))
                do
                    FL_PATH_OFFLOAD_IDX="$i1 $i2 $i3 $i4"

                    echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

                    ./main_fl \
                        $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
                        $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

                    sleep ${SLEEP_BETWEEN_ITER}
                done
            done
        done
    done

    ### PATH 5 ###

    FL_NUM_PATH=5

    for ((i1=0; i1<${fl_last_layer}; i1++))
    do
        for ((i2=0; i2<=${fl_last_layer}; i2++))
        do
            for ((i3=0; i3<=${fl_last_layer}; i3++))
            do
                for ((i4=0; i4<=${fl_last_layer}; i4++))
                do
                    for ((i5=0; i5<=${fl_last_layer}; i5++))
                    do
                        FL_PATH_OFFLOAD_IDX="$i1 $i2 $i3 $i4 $i5"

                        echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

                        ./main_fl \
                            $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
                            $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

                        sleep ${SLEEP_BETWEEN_ITER}
                    done
                done
            done
        done
    done

    ### PATH 6 ###

    FL_NUM_PATH=6

    for ((i1=0; i1<${fl_last_layer}; i1++))
    do
        for ((i2=0; i2<=${fl_last_layer}; i2++))
        do
            for ((i3=0; i3<=${fl_last_layer}; i3++))
            do
                for ((i4=0; i4<=${fl_last_layer}; i4++))
                do
                    for ((i5=0; i5<=${fl_last_layer}; i5++))
                    do
                        for ((i6=0; i6<=${fl_last_layer}; i6++))
                        do
                            FL_PATH_OFFLOAD_IDX="$i1 $i2 $i3 $i4 $i5 $i6"

                            echo "./main_fl   $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1   $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX"

                            ./main_fl \
                                $DNN $BATCH_SIZE $NUM_TILES $NUM_ITER $NUM_CORES $1 \
                                $FL_LAST_LAYER $FL_NUM_PATH $FL_PATH_OFFLOAD_IDX

                            sleep ${SLEEP_BETWEEN_ITER}
                        done
                    done
                done
            done
        done
    done


done





