#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}
TIME_OUT=${3:-"1800"}

echo "System: ${SYSTEM}"
echo "Number of GPUs: ${NUM_GPU}"
echo "GPU_SIZE: ${GPU_SIZE} GB"

source $(dirname $(realpath -L $0))/tasks.sh

main() {
    for task in "${!TASKS[@]}"; do
        if [[ "${task}" == "$TASK_NAME" ]] || [ "$TASK_NAME" == "all" ]; then
            echo "timeout -s SIGKILL $TIME_OUT bash ./benchmark_pytorch.sh $SYSTEM ${TASKS[${task}]} $task"
            timeout -s SIGKILL $TIME_OUT bash ./benchmark_pytorch.sh $SYSTEM ${TASKS[${task}]} $task

            echo "WAITING FOR CLEANUP ..."
            sleep 30
            echo "DONE"
        fi
    done

    chmod -R a+rwx /results/${SYSTEM}
}

main "$@"
