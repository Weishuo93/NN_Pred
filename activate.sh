# export NNPRED_BACKEND=ONNX
export NNPRED_BACKEND=TF
export PREDICTOR_HOME=${PWD}

export LD_LIBRARY_PATH=${PWD}/Predictor-Core/outputs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MY_TF_HOME=${PWD}/Predictor-Core/third_party/TF_libs
export MY_EIGEN_HOME=${PWD}/Predictor-Core/third_party/EIGEN_libs
export MY_ONNX_HOME=${PWD}/Predictor-Core/third_party/ONNX_libs

export LD_LIBRARY_PATH=${PWD}/Predictor-Core/third_party/TF_libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PWD}/Predictor-Core/third_party/ONNX_libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
