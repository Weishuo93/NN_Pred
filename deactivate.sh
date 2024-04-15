#!/bin/bash
# cd "${0%/*}"   # Run from this directory


script_absolute_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_absolute_path")

echo "Deactivating NNPred in directory: ${script_dir}"

unset MY_TF_HOME
unset MY_EIGEN_HOME
unset PREDICTOR_HOME

TF_LIB_DIR=${script_dir}/Predictor-Core/third_party/TF_libs/lib

if [[ "${LD_LIBRARY_PATH}" == "${TF_LIB_DIR}" ]]; then 
    LD_LIBRARY_PATH=""
fi

LD_LIBRARY_PATH=${LD_LIBRARY_PATH//":${TF_LIB_DIR}:"/":"} # delete any instances in the middle
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/#"${TF_LIB_DIR}:"/} # delete any instance at the beginning
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/%":${TF_LIB_DIR}"/} # delete any instance in the at the end


OUTPUTS_LIB=${script_dir}/Predictor-Core/src/outputs/lib


if [[ "${LD_LIBRARY_PATH}" == "${OUTPUTS_LIB}" ]]; then 
    LD_LIBRARY_PATH=""
fi


LD_LIBRARY_PATH=${LD_LIBRARY_PATH//":${OUTPUTS_LIB}:"/":"} # delete any instances in the middle
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/#"${OUTPUTS_LIB}:"/} # delete any instance at the beginning
LD_LIBRARY_PATH=${LD_LIBRARY_PATH/%":${OUTPUTS_LIB}"/} # delete any instance in the at the end 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

