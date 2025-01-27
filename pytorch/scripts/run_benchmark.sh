#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}
TIME_OUT=${3:-"1800"}
STAGE=${4}

RESULTS_DIR=/tmp/naic-benchmark-results-dir
SCRIPTS_DIR=$(dirname $(realpath -L $0))
DATA_DIR=

function system_info() {
    SYSTEM=${1:-"2080Ti"}
    CPU_NAME="$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g')"
    CPU_MEM="$(free -h | grep Mem: | awk '{ print $2 }')"

    GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
    GPU_NAME="${GPU_NAME// /_}"
    GPU_MEM="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader)"
    GPU_MEM="${GPU_MEM// /_}"

    NVIDIA_DRIVER="$(nvidia-smi | grep "Driver Version:" | awk '{ print $3 }')"
    CUDA_VERSION="$(nvcc --version | grep release | awk '{ print $NF }')"

    CUDNN_MAJOR="$(cat /usr/include/cudnn.h | grep "#define CUDNN_MAJOR" | awk '{ print $NF }')"
    CUDNN_MINOR="$(cat /usr/include/cudnn.h | grep "#define CUDNN_MINOR" | awk '{ print $NF }')"
    CUDNN_PATCHLEVEL="$(cat /usr/include/cudnn.h | grep "#define CUDNN_PATCHLEVEL" | awk '{ print $NF }')"
    CUDNN_VERSION=${CUDNN_MAJOR}"."${CUDNN_MINOR}"."${CUDNN_PATCHLEVEL}

    MB="$(cat /sys/devices/virtual/dmi/id/board_{vendor,name,version} | tr '\n' ' ')"
    PLATFORM_NAME="$(cat /etc/os-release | grep "PRETTY_NAME=" | cut -c 14- | rev | cut -c 2- | rev)"
    PT_VERSION="$(python -c "import torch; print(torch.__version__)" | awk '{ print $NF }')"
    RESULTS_PATH=$2/${SYSTEM}

    mkdir -p $RESULTS_PATH

    SYSTEM_FILE=${RESULTS_PATH}/sys_pytorch.txt

    echo "CPU: "${CPU_NAME}                >> $SYSTEM_FILE
    echo "CPU Memory: "${CPU_MEM}          >> $SYSTEM_FILE
    echo "GPU: "${GPU_NAME}                >> $SYSTEM_FILE
    echo "GPU Memory: "${GPU_MEM}          >> $SYSTEM_FILE
    echo "NVIDIA driver: "${NVIDIA_DRIVER} >> $SYSTEM_FILE
    echo "CUDA Version: "${CUDA_VERSION}   >> $SYSTEM_FILE
    echo "CUDNN Version: "$CUDNN_VERSION   >> $SYSTEM_FILE
    echo "Motherboard: "${MB}              >> $SYSTEM_FILE
    echo "OS: "${PLATFORM_NAME}            >> $SYSTEM_FILE
    echo "PyTorch Version: "${PT_VERSION}  >> $SYSTEM_FILE
    chmod -R a+rwx $SYSTEM_FILE
}

function install_requirements() {
    # FIXME: when using torchrun from the system install then
    # boto3 seems not to be picked up correctly
    pip install termcolor boto3
    pip install 'git+https://github.com/NVIDIA/dllogger'

    # bc is used for the scaling
    # see computation scripts/config_v1/config_pytorch_1GB.sh
    apt update && apt install -y bc vim

    echo "Patching benchmarks"
    if [ ! -d benchmark ]; then
        echo "benchmark folder does not exist. Did you miss to link the DeepLearningExamples in $PWD"
        exit 10
    fi

    cp $SCRIPTS_DIR/patch/run_squad.py benchmark/LanguageModeling/BERT
    cp $SCRIPTS_DIR/patch/multiproc.py benchmark/SpeechSynthesis/Tacotron2
}

function compile_results() {
    echo "Validating and compiling results in $RESULTS_DIR"
    # ensure that pandas is available
    python3 -c "import pandas"
    if [ $? -ne 0 ]; then
        uv pip install --system pandas
    fi
    python $SCRIPTS_DIR/check.py --path $RESULTS_DIR/${SYSTEM} |& tee $RESULTS_DIR/${SYSTEM}/summary.txt
    python $SCRIPTS_DIR/compile_results_pytorch_v2.py --path $RESULTS_DIR --output-path $RESULTS_DIR
}

function get_task_folder() {
    source $SCRIPTS_DIR/tasks.sh
    source $SCRIPTS_DIR/config/config_pytorch_1GB.sh $NUM_GPU $GPU_SIZE $GPU_DEVICE_TYPE

    TASK=$1
    TASK_PARAMS=${TASK}_PARAMS[@]
    command_path=$(sed 's/\.*args.*//' <<<${!TASK_PARAMS})
    echo $(realpath -L ${SCRIPTS_DIR}/${command_path})
}

function get_task_arguments() {
    source $SCRIPTS_DIR/tasks.sh
    source $SCRIPTS_DIR/config/config_pytorch_1GB.sh $NUM_GPU $GPU_SIZE $GPU_DEVICE_TYPE

    TASK=$1
    TASK_PARAMS=${TASK}_PARAMS[@]
    command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    echo $command_para
}

function prepare_task_venv() {
    TASK=$1
    TASK_FOLDER=$2

    TASK_VENV_NAME=venv-$(uname -i)-$TASK
    python -m venv --system-site-packages $TASK_VENV_NAME
    source $TASK_VENV_NAME/bin/activate

    echo "Installing requirements for benchmark: $TASK_FOLDER/requirements.txt for $(which python)"
    pip install -U setuptools wheel build
    pip install -r $TASK_FOLDER/requirements.txt
    pip list
}

function run_tasks() {
    echo "System: ${SYSTEM}"
    echo "Number of GPUs: ${NUM_GPU}"
    echo "GPU_SIZE: ${GPU_SIZE} GB"

    cd $SCRIPTS_DIR
    source ./tasks.sh

    for task in "${!TASKS[@]}"; do
        if [[ "${task}" == "$TASK_NAME" ]] || [ "$TASK_NAME" == "all" ]; then
            # LOAD VENV
            TASK_VENV_NAME=venv-$(uname -i)-$task
            if [ ! -d $TASK_VENV_NAME ]; then
                echo "virtual env: $TASK_VENV_NAME does not exist, please prepare image first"
                exit 20
            fi

            source $TASK_VENV_NAME/bin/activate
            echo "Using venv: $(which python)"

            RESULTS_PATH=$RESULTS_DIR/${SYSTEM}/${task}
            mkdir -p $RESULTS_PATH
            rm ${RESULTS_PATH}/*.txt

            echo "timeout -s SIGKILL $TIME_OUT bash ./benchmark_pytorch.sh $SYSTEM ${TASKS[${task}]} $task $RESULTS_PATH $NUM_EXP"
            timeout -s SIGKILL $TIME_OUT bash ./benchmark_pytorch.sh $SYSTEM ${TASKS[${task}]} $task $RESULTS_PATH $NUM_EXP

            # deactivate venv
            echo -n "Deactivating venv $(which python) -- "
            deactivate
            echo " python now: $(which python)"

            echo "WAITING FOR CLEANUP ..."
            sleep 30
            echo "DONE"
        fi
    done

    chmod -R a+rwx $RESULTS_DIR/${SYSTEM}
}

function prepare_venvs() {
    install_requirements

    cd $SCRIPTS_DIR
    source ./tasks.sh

    for task in "${!TASKS[@]}"; do
        if [[ "${task}" == "$TASK_NAME" ]] || [ "$TASK_NAME" == "all" ]; then
            TASK_FOLDER=$(get_task_folder $task)
            echo "TASK_FOLDER $TASK_FOLDER"
            prepare_task_venv $task $TASK_FOLDER
        fi
    done

    chmod -R a+rwx $RESULTS_DIR/${SYSTEM}
}

function run_benchmarks() {
    if [ ! -d /scripts ]; then
        echo "It seems this script is not run inside a docker container"
    fi
    
    echo "Preparing system configuration for $SYSTEM"
    system_info $SYSTEM $RESULTS_DIR
    
    echo "Running benchmarks: task(s): $TASK_NAME -- timeout: $TIME_OUT"
    run_tasks $TASK_NAME

    compile_results
}

if [ $RUN_BENCHMARK_ONLY -ne 1 ]; then
    prepare_venvs
fi

if [ $PREPARE_BENCHMARK_ONLY -ne 1 ]; then
    run_benchmarks
fi


