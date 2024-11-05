#!/usr/bin/env bash
export HGCN_HOME=$(pwd)
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export ROOT_DIR="$(dirname "$HGCN_HOME")"
export DATAPATH="$ROOT_DIR/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64 # replace by appropiate cuda version

source ../hgcn/bin/activate # replace with source activate hgcn if used conda