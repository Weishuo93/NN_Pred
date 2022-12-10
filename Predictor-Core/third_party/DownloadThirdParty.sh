EIGEN_VER=3.4.0
TF_VER=2.6.0
ONNX_VER=1.13.1

function get_eigen {
    EIGEN_URL=https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz
    mkdir -p ./EIGEN_libs
    wget ${EIGEN_URL} -P ./
    tar -xzf ./eigen-${EIGEN_VER}.tar.gz --strip-components=1 -C ./EIGEN_libs 
}

function get_tf {
    TF_URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${TF_VER}.tar.gz
    mkdir -p ./TF_libs
    wget ${TF_URL} -P ./
    tar -xzf ./libtensorflow-cpu-linux-x86_64-${TF_VER}.tar.gz -C ./TF_libs 
}

function get_onnx {
    ONNX_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-linux-x64-${ONNX_VER}.tgz
    mkdir -p ./ONNX_libs
    wget ${ONNX_URL} -P ./
    tar -xzf ./onnxruntime-linux-x64-${ONNX_VER}.tgz --strip-components=1 -C ./ONNX_libs 
}


get_eigen
get_tf
get_onnx    

