#!/bin/bash

SYSTEM=${1:-"2080Ti"}
func=${2:-"benchmark_pytorch_ncf"}
task=${3:-"PyTorch_ncf_FP32"}

source config_v1/config_pytorch_1GB.sh $NUM_GPU $GPU_SIZE
get_current_time() {
    echo $(TZ=UTC date +"%s")
}

run_it() {
    start_time=$(get_current_time)
    $@
    end_time=$(get_current_time)

    result_marker="SUCCESS"
    if grep -E "RuntimeError|OutOfMemoryError|NameError|ImportError" "$result"; then
        result_marker="FAILURE"
    fi
    echo "# BEGIN SUMMARY" >> ${result}
    echo "# TASK START END EXIT_CODE SLURM_JOB_ID" >> ${result}
    echo "${task} ${start_time} ${end_time} ${result_marker} ${SLURM_JOB_ID}" >> ${result}
    echo "# END SUMMARY" >> ${result}
}

prepare_requirements() {
    if [ "$(command -v uv)" == "" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
    fi

    uv pip install --system -r requirements.txt
    pip list
}


benchmark_pytorch_ssd() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--batch-size )\w+'`

    prepare_requirements

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH $((BATCH * NUM_GPU))" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it torchrun --nproc_per_node=${NUM_GPU} main.py \
    --mode benchmark-training ${command_para} |& tee ${result} 
}


benchmark_pytorch_resnet50() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--batch-size )\w+'`

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH $((BATCH * NUM_GPU))" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it python ./multiproc.py --nproc_per_node ${NUM_GPU} ./main.py \
    ${command_para} |& tee ${result}
}


benchmark_pytorch_maskrcnn() {

    echo "Skip MaskRCNN until maskrcnn_benchmark can be built."
    return 1

    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=SOLVER.IMS_PER_BATCH )\w+'`

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH ${BATCH}" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # python setup.py install
    # pip install -r requirements.txt

    # export NCCL_P2P_DISABLE=1
    run_it torchrun --nproc_per_node=${NUM_GPU} --use_env tools/train_net.py \
    --skip-test \
    ${command_para} \
    | tee $result
    
    time=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
    statement=`cat $result | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1`
    calc=$(echo $time 1.0 $GLOBAL_BATCH | awk '{ printf "%f", $2 * $3 / $1 }')
    
    echo "Training perf is: "$calc" FPS" >> ${result}
    rm /results/*.txt
    rm /results/*.pth
    rm /results/*checkpoint* 
}


benchmark_pytorch_gnmt() {

    local task="$1"
    local result="$2"

    prepare_requirements

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--train-batch-size )\w+'`
    
    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH $((BATCH * NUM_GPU))" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it torchrun --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}
}


benchmark_pytorch_ncf() {
    
    local task="$1"
    local result="$2"

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--batch_size )\w+'`

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH ${BATCH}" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it torchrun --nproc_per_node=${NUM_GPU} ncf.py ${command_para} |& tee ${result}
}


benchmark_pytorch_transformerxl() {
    
    local task="$1"
    local result="$2"

    prepare_requirements
    
    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--batch_size )\w+'`

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH ${BATCH}" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it torchrun --nproc_per_node=${NUM_GPU} train.py ${command_para} |& tee ${result}
}


benchmark_pytorch_tacotron2() {
    
    local task="$1"
    local result="$2"

    prepare_requirements

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=`echo ${!TASK_PARAMS} | grep -oP '(?<=--batch-size )\w+'`

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH $((BATCH * NUM_GPU))" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para
    echo "************************************************************"

    # export NCCL_P2P_DISABLE=1
    run_it python -m multiproc ${NUM_GPU} train.py \
    ${command_para}  |& tee ${result}
}


benchmark_pytorch_bert_squad() {

    local task="$1"
    local result="$2"
    
    prepare_requirements

    TASK_PARAMS=${task}_PARAMS[@]
    local command_para=$(sed 's/.*args //' <<<${!TASK_PARAMS})
    local BATCH=${task}_PARAMS[4]

    echo "************************************************************"
    echo $command_para
    echo "GLOBAL_BATCH $((BATCH * NUM_GPU))" > ${RESULTS_PATH}benchmark.para
    echo "GPU ${NUM_GPU}" >> ${RESULTS_PATH}benchmark.para    
    echo "************************************************************"

    run_it bash scripts/run_squad.sh ${command_para} |& tee ${result}
}


echo "${task} started: "

RESULTS_PATH=/results/${SYSTEM}/${task}/
TASK_PARAMS=${task}_PARAMS[@]
MONITOR_INTERVAL=2

command_path=$(sed 's/\.*args.*//' <<<${!TASK_PARAMS})

mkdir -p $RESULTS_PATH

pushd .
cd $command_path

rm ${RESULTS_PATH}*.txt

for i in $(seq 1 $NUM_EXP); do
    name=${RESULTS_PATH}$(date +%d-%m-%Y_%H-%M-%S)
    file_result="${name}.txt"
    $func $task $file_result
    sleep 5
done

chmod -R a+rwx $RESULTS_PATH
echo "${task} ended."
popd 

