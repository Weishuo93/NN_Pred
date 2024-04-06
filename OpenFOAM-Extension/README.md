# OpenFOAM-Extension
Application of the **NN_Pred** in **OpenFOAM**. 

The folder consists of the `TF_OF_Predictor` and the tutorial cases including a `ML Laplacian Solver` and a `ML-RANS Model`. The `TF_OF_Predictor` handles the data I/O with `volScalarField` or `scalarField` data type in OpenFOAM.

## Installation of TF_OF_Predictor
Firstly, the `Predictor-Core` needs to be properly compiled. Then the `OpenFOAM-Extension` can be compiled. The entire folder supports the following OpenFOAM versions.

For OpenFOAM.org versions:
*   [**OpenFOAM v4.x**](https://github.com/OpenFOAM/OpenFOAM-4.x) or
*   [**OpenFOAM v5.x**](https://github.com/OpenFOAM/OpenFOAM-5.x) or
*   [**OpenFOAM v6**](https://github.com/OpenFOAM/OpenFOAM-6)

For OpenFOAM.com versions (should work for all distributions since v1606):
*   [**latest (v2312)**](https://develop.openfoam.com/Development/openfoam) 

The `TF_OF_Predictor` can be compiled in more recent version.

In an activated `wmake` envirment, the compilation of the `TF_OF_Predictor` can be done by:
```sh
cd OpenFOAM-Extension  # Go to the OF Extension dir

# build OpenFOAM extension and test program
# this requires an installation of Openfoam, with 'wmake' script available
make ofso    # OpenFOAM API
make oftest  # OpenFOAM test program without running
make runof   # OpenFOAM test program and run
```

By default, the `TF_OF_Predictor` binary library can be found in `$FOAM_USER_LIBBIN` under name `libTF_OF_Predictor.so`.

## Usage
The predictor will be created by parsing the information in an OpenFOAM dictionary, within the dictionary you should specify all the needed keys with legal values to load the TF model.

```c++
model_pb   //your model name, default is "model"    
{
    readFromPB     yes;     //yes for pb format, no for SavedModel format
    modelDirectory "path/to/your/model";
    tags           "serve";  //default tag for SavedModel is "serve", will be activated only when "readFromPB" is "no".
    copyMethod     Eigen;           // Eigen for fast, Simple for stable in case for unpredictable bugs
    layout         RowMajor;        // ColMajor or RowMajor
    inputs         ("your_input_1" "your_input_2" ...);       //Node names for inputs, use space to separate multiple inputs
    outputs        ("your_output_1" "your_output_2" ...);     //Node names for outputs, use space to separate multiple outputs
}
```
In OpenFOAM program where you want to use the predictor, the basic usage is: 
```c++
#include "TF_OF_Predictor.H"   //headers containing all the class definition and member functions
//... some codes ... 

// Construct predictor
TF_OF_Predictor pd = TF_OF_Predictor("your_predictor_dictionary", "your_model_name");
// For example, if your dictionary is saved in constant/ folder with filename TF_Predictor_Dict, then 
// the predictor should be construct with TF_OF_Predictor("constant/TF_Predictor_Dict", "your_model_name");

// Prepare the data container:
// The predictor can take List of pointers of scalarField or volScalarField. Pass the fields' reference to the List
// For example, your model has two input nodes and three output nodes, then you need to create Foam::List object.
Foam::List<Foam::List<scalarField*>> multi_inputs(2);  //2 input nodes
Foam::List<Foam::List<scalarField*>> multi_outputs(3);  //3 output nodes
// If the first input node takes tensor shape with [-1, 3], means it requires 3 features, so append the field pointer
// to the container:
multi_inputs[0].append(feature1_field);
multi_inputs[0].append(feature2_field);
multi_inputs[0].append(feature3_field);
...
// Attention that the memory of a field is continuously mapped in OpenFOAM, but a [-1, 3] shaped tensor mapped each 
// row of 3 elements together, therefore layout transpose is needed in the memory, the "layout" key in the dictionary
// should be specified with "ColMajor".
// The "feature1_field" shoule be constructed by before passing to the List: 
Foam::scalarField feature1_field;
// or
Foam::volScalarField feature1_field;

// run the model:
pd.predict(multi_inputs, multi_outputs);
// the fields in outputs should be updated with new output values
```

Please see more details with the test program in *TF_OF_Predictor/test/OpenFOAM_test_AplusB.C*


## ML-Laplacian Solver 
This solver is to integrate a ML radiation relation to a heat transfer problem. The problem is described in [this paper](https://arxiv.org/abs/2209.12339) under tutorial 1. 

To compile the solvers in tutorial cases, one can execute:
```sh
cd tutorials/heat_transfer/Solvers/laplacianFoam_Ideal  # Laplacian solver for known analytical emissivity
wmake # Build the solver
```

```sh
cd tutorials/heat_transfer/Solvers/laplacianFoam_ML  # Laplacian solver for machine-learning emissivity
wmake # Build the solver
```
The compilation will create two executables: `laplacianFoam_Ideal` and `laplacianFoam_ML` in the `$FOAM_USER_APPBIN` directory. To run the heat transfer simulation, go the the `Cases` folder and run the corresponding solver.

```sh
cd tutorials/heat_transfer/Cases/Stick_Ideal # Go to the analytical-emissivity heat transfer case
./Allrun # Run the simulation
```
```sh
cd tutorials/heat_transfer/Cases/Stick_ML # Go to ML-emissivity heat transfer case
./Allrun # Run the simulation
```
The Python notebook to post process the simulation results are provided as: `tutorials/heat_transfer/Cases/PostProcess.ipynb`

The script to train the ML model is provided, the file path is: `tutorials/heat_transfer/Cases/Training.ipynb`

## ML-RANS Turbulence Model 
This ML turbulence model library is to achieve the methodology of the [ML-RANS framework](https://www.sciencedirect.com/science/article/pii/S0142727X21000527). The problem is described in [this paper](https://arxiv.org/abs/2209.12339) under tutorial 2. 

To compile the solvers in tutorial cases, one can execute:
```sh
cd tutorials/turbulent_channel/Solvers/ML_RAS_Model_Incompressible  # ML_RANS model for incompressible flow
wmake # Build the turbulence model library
```
```sh
cd tutorials/turbulent_channel/Solvers/ML_RAS_Model_Compressible  # ML_RANS model for compressible flow
wmake # Build the solver
```
The compilation will create two binary libraries: `libkOmegaSSTML_Incompressible.so` and `libkOmegaSSTML_Compressible.so` in the `$FOAM_USER_LIBBIN` directory. To run the turbulent channel simulation, go the the `Cases` folder and run the `simpleFoam` solver.
```sh
cd tutorials/turbulent_channel/Cases/Re_180_kwSST # Go to the original kwSST case
simpleFoam >log &   # Run the simulation
```
```sh
cd tutorials/turbulent_channel/Cases/Re_180_ML # Go to the ML-RANS case
simpleFoam >log &   # Run the simulation
```
```sh
cd tutorials/turbulent_channel/Cases/Re_180_kwSST_2blk # Go to the original kwSST parallel case
decomposePar
mpiexec -np 2 simpleFoam -parallel >log &   # Run the parallel simulation
```
```sh
cd tutorials/turbulent_channel/Cases/Re_180_ML_2blk # Go to the ML-RANS parallel case
decomposePar
mpiexec -np 2 simpleFoam -parallel >log &   # Run the parallel simulation
```

To observe the residual curve, one can use the provided gnuplot script:
```sh
gnuplot Residuals
```



