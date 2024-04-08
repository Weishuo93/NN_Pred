#!/bin/bash

# Set default backend to TF, can be overridden by user
export NNPRED_BACKEND=${NNPRED_BACKEND:-TF}
# export NNPRED_BACKEND=${NNPRED_BACKEND:-ONNX}

export PREDICTOR_HOME=${PWD}

# Set default library paths if not already defined by the user
export MY_EIGEN_HOME=${MY_EIGEN_HOME:-"${PWD}/Predictor-Core/third_party/EIGEN_libs"}
export MY_TF_HOME=${MY_TF_HOME:-"${PWD}/Predictor-Core/third_party/TF_libs"}
export MY_ONNX_HOME=${MY_ONNX_HOME:-"${PWD}/Predictor-Core/third_party/ONNX_libs"}

# Update LD_LIBRARY_PATH with the specified or default library paths
export LD_LIBRARY_PATH=${MY_TF_HOME}/lib:${MY_ONNX_HOME}/lib:${PWD}/Predictor-Core/outputs/lib:${LD_LIBRARY_PATH}