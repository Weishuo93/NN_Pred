# NN_Pred -- An interfacing library to deploy NN models in CFD codes

This library aims at running tensorflow models in C++ and Fortran program. The library is mainly designed for deploying neural networks in CFD softwares, the extensions in **OpenFOAM** and **CFL3D** might reduce the burden in integrating ML with CFD softwares.

The following features might be useful to you:
- Support loading *.pb graph saved by frozen-graph utils in TF1
- Support loading [SavedModel](https://www.tensorflow.org/guide/saved_model) format saved by TF2 Keras and TF estimator
- Support memory layout transpose when set/get data to/from nodes
- Automatic type casting according to the type of your data container and node definition
- Vectorizatized casting and transpose is integrated through Eigen 
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
The `Predictor-Core` uses Tensorflow C-API, the dependencies are as follow:

- [libtensorflow.so](https://www.tensorflow.org/install/lang_c)
- [Eigen](https://gitlab.com/libeigen/eigen/-/releases)

By executing the download scripts they will be downloaded into third_party directory:
```sh
cd Predictor-Core/third_party
./DownloadThirdParty.sh
```

If you want to use the locally installed libs, please modify the two environment variables set in the `activate.sh` to point at the locally installed path:
```sh
export MY_EIGEN_HOME=your/eigen/dir
export MY_TF_HOME=your/TF/dir/with/libtensorflow.so/
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

After the compilation, you might need to add the compiled libraries (located in `./Predictor-Core/outputs/lib`) in your `$LD_LIBRARY_PATH`. This operation is done by sourcing the `activate.sh` in the beginning of the tutorial.

The installation of `OpenFOAM-Extension` and `CFL3D-Extension` please refer to the `README.md` in each separated folders.

## How to cite
If this software brings convenience to you, please consider citing the following paper, reported in the bibtex entries:
```
@article{

}
```

## License
The source files in `Predictor-Core` folder is freely available under the MIT License. 

The files in `OpenFOAM-Extension` folder is published under GNU Lesser General Public License (LGPL), Version 3. The files in `CFL3D-Extension` folder is published under Apache License, Version 2.






