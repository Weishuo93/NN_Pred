mkdir -p ./TF_libs
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz -P ./
tar -C ./TF_libs -xzf ./libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz

wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz -P ./
tar -C ./ -xzf ./eigen-3.3.9.tar.gz
