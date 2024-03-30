# NN_Pred -- An interfacing library to deploy ML models in CFD codes

This library aims at running ML models in C++ and Fortran programs. The library is mainly designed for deploying neural networks in CFD software, the extensions in **OpenFOAM** and **CFL3D** might reduce the burden of integrating ML with CFD software.

The following features might be useful to you:
- TensorFlow Backends: Support loading *.pb graph and [SavedModel](https://www.tensorflow.org/guide/saved_model) format.
- ONNX Backends: Support loading [*.onnx](https://onnxruntime.ai/) ML models (from PyTorch, Sklearn, etc)
- Support memory layout transpose when set/get data to/from nodes
- Automatic type casting according to the type of your data container and node definition
- Vectorizatized casting and transposing is integrated through Eigen 
- Basic APIs in C++ and Fortran
- Extensions in OpenFOAM and CFL3D to assist CFD simulation

### A minimal example of usage of the core predictor in C++:

```c++
// Load the Graph:
Predictor pd("graph.pb");

// Register the nodes in graph that you want to set input data in or get the output data
pd.regist_node("your_input_name", Predictor::INPUT_NODE);
pd.regist_node("your_output_name", Predictor::OUTPUT_NODE);

// If the tensorshape in your nodes is with unknown dimension, e.g., [-1, 3]
// set the unknown dimension with exact number of examples you want to predict 
pd.set_data_count(n_data);

// Create the input data to feed
std::vector<float> data_in({/* some data */});

// Feed data to the inputs
pd.set_node_data("your_input_name", data_in);

// Run the model
pd.run();

// Create a container to hold output data, suppose the shape of "your_output_name" is [-1, 3, 2], then 
int n_known_dim = 3 * 2;
std::vector<float> data_out(n_data * n_known_dim); 

// Retrieve data from the outputs
pd.get_node_data("your_output_name", data_out)

```


## How to install
The first step is to activate this folder by executing the script:
```sh
source activate.sh
```
This script sets the necessary environmental variables to properly compile the library. The `Predictor-Core` needs to be compiled firstly, as the `OpenFOAM-Extension` and `CFL3D-Extension` all depends on the core predictor.

### Third party dependencies
The `Predictor-Core` supports two different backends, The TF backends rely on Tensorflow C-API, the dependencies are as follows:
- [libtensorflow.so](https://www.tensorflow.org/install/lang_c)
- [Eigen](https://gitlab.com/libeigen/eigen/-/releases)

The dependency of ONNX backends is:
 - [OnnxRuntime](https://onnxruntime.ai/)
 - [Eigen](https://gitlab.com/libeigen/eigen/-/releases)

By executing the download scripts they will be downloaded into third_party directory:
```sh
cd Predictor-Core/third_party
./DownloadThirdParty.sh
```
### Switching backends
The backend is specified by an environmental variable in the activate.sh file. One can modify the script or set the environment variable manually in the shell command line.
```sh
export NNPRED_BACKEND=ONNX  # ONNX Runtime backend
export NNPRED_BACKEND=TF    # TensorFlow backend
```



### Build the Predictor-Core

```sh
# Go to the source file dir:
cd Predictor-Core

# build the core predictor and test program 
make cxxso    # the core predictor in C++ 
make cxxtest  # C++ test program without running
make run      # C++ test program and run 

# build fortran extension and test program
make f90so    # fortran API
make f90test  # fortran test program without running
make runf     # fortran test program and run
```

After the compilation, both the two backends are compiled by default, and the make target: `cxxso` will create a symbolic link pointing to the library specified by environmental variable `$NNPRED_BACKEND`. If the backend needs to be changed, one can modify `$NNPRED_BACKEND` and execute the make target:
```sh
make alias-predictor # Create the symbolic link to the backend lib
```

you might need to add the compiled libraries (located in `./Predictor-Core/outputs/lib`) in your `$LD_LIBRARY_PATH`. This operation is done by sourcing the `activate.sh` at the beginning of the tutorial.

The installation of `OpenFOAM-Extension` and `CFL3D-Extension` please refer to the `README.md` in each separate folder.
 
### [Build OpenFOAM-Extension](https://github.com/Weishuo93/NN_Pred/tree/master/OpenFOAM-Extension)

### [Build CFL3D-Extension](https://github.com/Weishuo93/NN_Pred/tree/master/CFL3D-Extension)

### Locally installed third-party libs
If you want to use the locally installed libs, please modify the two environment variables set in the `activate.sh` to point at the locally installed path:
```sh
export MY_EIGEN_HOME=your/eigen/dir
export MY_TF_HOME=your/TF/dir/with/libtensorflow.so
export MY_ONNX_HOME=your/ONNX/runtime/library/dir
```

## API Documents

The API are all demonstrated with a simple `A+B=C` model, the PDF files can be found as follows:
- [C++ example and API document](https://github.com/Weishuo93/NN_Pred/blob/master/Documentation/Cpp_Example_API.pdf)
- [Fortran example and API document](https://github.com/Weishuo93/NN_Pred/blob/master/Documentation/Fortran_Example_API.pdf)
- [OpenFOAM example and API document](https://github.com/Weishuo93/NN_Pred/blob/master/Documentation/OpenFOAM_Module_Example_API.pdf)
- [Explanation on CFL3D ML module](https://github.com/Weishuo93/NN_Pred/blob/master/Documentation/Fortran_Module_Template.pdf)


## How to cite
If this software brings convenience to you, please consider citing the following paper, reported in the bibtex entries:
```
@article{NNPred,
  title={NNPred: Deploying neural networks in computational fluid dynamics codes to facilitate data-driven modeling studies},
  author={Liu, Weishuo and Song, Ziming and Fang, Jian},
  journal={Computer Physics Communications},
  pages={108775},
  year={2023},
  publisher={Elsevier}
}
```

## License
The source files in the `Predictor-Core` folder are freely available under the MIT License. 

The files in the `OpenFOAM-Extension` folder are published under GNU Lesser General Public License (LGPL), Version 3. The files in the `CFL3D-Extension` folder are published under Apache License, Version 2.






