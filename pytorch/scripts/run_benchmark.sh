#!/bin/bash

SYSTEM=${1:-"2080Ti"}
TASK_NAME=${2:-"all"}
TIME_OUT=${3:-"1800"}

RESULTS_DIR=/results
SCRIPTS_DIR=/scripts

if [ ! -d /scripts ]; then
    echo "It seems this script is not run inside a docker container"
fi

# bc is used for the scaling
# see computation scripts/config_v1/config_pytorch_1GB.sh
apt update && apt install -y bc

pip install termcolor
pip install 'git+https://github.com/NVIDIA/dllogger'

echo "Patching benchmarks"
cp $SCRIPTS_DIR/patch/run_squad.py benchmark/LanguageModeling/BERT
cp $SCRIPTS_DIR/patch/multiproc.py benchmark/SpeechSynthesis/Tacotron2

echo "Preparing pip packages"
if [[ "${TASK_NAME}" == *"ssd"* ]] || [ $TASK_NAME = "all" ] || [[ "${TASK_NAME}" == *"maskrcnn"* ]]; then
	pip install -r benchmark/Detection/SSD/requirements.txt
fi

echo "Preparing system configuration for $SYSTEM"
./run_system_pytorch.sh $SYSTEM

echo "Running benchmarks: task(s): $TASK_NAME -- timout: $TIME_OUT"
./run_benchmark_pytorch.sh $SYSTEM $TASK_NAME $TIME_OUT

echo "Validating and compiling results in $RESULTS_DIR"
python $SCRIPTS_DIR/check.py --path $RESULTS_DIR/${SYSTEM} |& tee $RESULTS_DIR/${SYSTEM}/summary.txt
for precision in "fp32" "fp16"; do
    python $SCRIPTS_DIR/compile_results_pytorch_v2.py --path $RESULTS_DIR --output-path $RESULTS_DIR --precision $precision
done

