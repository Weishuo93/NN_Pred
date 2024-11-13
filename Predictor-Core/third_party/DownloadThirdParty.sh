#!/bin/bash

# Define default versions
EIGEN_VER=3.4.0
TF_VER=2.6.0
ONNX_VER=1.12.1

# Function to download and extract third-party libraries
function download_and_extract {
    local url=$1
    local tar_name=$2
    local lib_home=$3
    local strip_components=$4

    echo "Checking ${lib_home} for ${tar_name}..."
    if [ ! -d "${lib_home}" ] || [ "$(ls -A ${lib_home})" == "" ]; then
        echo "Downloading and extracting ${tar_name} from ${url}..."
        wget -q --show-progress "${url}" -O "${tar_name}"
        if [ $? -ne 0 ]; then
            echo "Error downloading ${tar_name}."
            echo "Please download it manually from: ${url}"
            exit 1
        fi
        mkdir -p "${lib_home}"
        tar -xzf "${tar_name}" --strip-components=${strip_components} -C "${lib_home}"
    else
        echo "${lib_home} already exists and is not empty. Skipping download."
    fi
}

# Attempt to download and extract each third-party library
download_and_extract \
    "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz" \
    "eigen-${EIGEN_VER}.tar.gz" \
    "${MY_EIGEN_HOME:-"${PWD}/Predictor-Core/third_party/EIGEN_libs"}" \
    1

download_and_extract \
    "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${TF_VER}.tar.gz" \
    "libtensorflow-cpu-linux-x86_64-${TF_VER}.tar.gz" \
    "${MY_TF_HOME:-"${PWD}/Predictor-Core/third_party/TF_libs"}" \
    0

download_and_extract \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-linux-x64-${ONNX_VER}.tgz" \
    "onnxruntime-linux-x64-${ONNX_VER}.tgz" \
    "${MY_ONNX_HOME:-"${PWD}/Predictor-Core/third_party/ONNX_libs"}" \
    1

echo "All required libraries have been checked and downloaded if necessary."
