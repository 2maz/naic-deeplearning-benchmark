#!/bin/bash

function scale() {
    if [ -n $GPU_SIZE ]; then
        echo "ERROR: GPU_SIZE is not set"
        exit 101
    fi

    if [ -n $GPU_COUNT ]; then
        echo "ERROR: GPU_COUNT is not set"
        echo 102
    fi

    if [ -n $GPU_DEVICE_TYPE ]; then
        echo "WARNING: GPU_DEVICE_TYPE is not set"
    fi

    # 1rst argument is base-batchsize for 1GB
    # 2nd argument is an additional scaling introduced for multiple GPUs 
    #  to avoid out-of-memory errors
    SCALING_FACTOR=$GPU_SIZE
    if [ "$GPU_COUNT" -gt 1 ]; then
        if [ "$2" != "" ]; then
            export MULTIPLE_GPUS="*2*$2"
        fi
    fi
    if [ "$GPU_DEVICE_TYPE" == 'xpu' ]; then
        VENDOR_SCALE_FACTOR="*0.85"
    fi
    NUMBER=`echo "$1*${SCALING_FACTOR}${MULTIPLE_GPUS}${VENDOR_SCALE_FACTOR}" | bc`
    echo ${NUMBER%.*}
}
