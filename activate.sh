#!/bin/bash
# cd "${0%/*}"   # Run from this directory

# Set default backend to TF, can be overridden by user
export NNPRED_BACKEND=${NNPRED_BACKEND:-TF}
# export NNPRED_BACKEND=${NNPRED_BACKEND:-ONNX}

script_absolute_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_absolute_path")

echo "Using NNPred in directory: $script_dir"

export PREDICTOR_HOME=${script_dir}


# Set default library paths if not already defined by the user
export MY_EIGEN_HOME=${MY_EIGEN_HOME:-"${PREDICTOR_HOME}/Predictor-Core/third_party/EIGEN_libs"}
export MY_TF_HOME=${MY_TF_HOME:-"${PREDICTOR_HOME}/Predictor-Core/third_party/TF_libs"}
export MY_ONNX_HOME=${MY_ONNX_HOME:-"${PREDICTOR_HOME}/Predictor-Core/third_party/ONNX_libs"}

# Update LD_LIBRARY_PATH with the specified or default library paths
export LD_LIBRARY_PATH=${MY_TF_HOME}/lib:${MY_ONNX_HOME}/lib:${PREDICTOR_HOME}/Predictor-Core/outputs/lib:${LD_LIBRARY_PATH}