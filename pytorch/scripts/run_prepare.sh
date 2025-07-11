#!/bin/bash
set -e

NAME_DATASET=${1:-"all"}
DATA_DIR=${2:-/data}
BENCHMARK_DIR=${3:-benchmark}

SCRIPT_DIR=$(realpath -L $(dirname $0))

DEBIAN_FRONTEND=noninteractive apt update && apt install -y --quiet \
    curl \
    git \
    python3-venv python3-pip \
    unzip \
    wget

# Run all (matching) preparation script found in the subfolder prepare.d
for file in $(ls $SCRIPT_DIR/conf.d/*.prepare); do
    DATASET_NAME=$(basename -s .prepare $file)
    if [ "$DATASET_NAME" == "$NAME_DATASET" ] || [ "$NAME_DATASET" == 'all' ]; then
        echo "Running: $file $DATA_DIR $BENCHMARK_DIR"
        $file $DATA_DIR $BENCHMARK_DIR
    fi
done
