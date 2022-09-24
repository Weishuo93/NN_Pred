# Core Predictor of NN_Pred
The predictor consists of a C++ predictor class and the Fortran APIs.

## Usage

### Create tensorflow models
This predictor support two types of model formats created by tensorflow.
- Frozen graph under *.pb file 
- [SavedModel format](https://www.tensorflow.org/guide/saved_model) 

Both formats have examples to create tensorflow models. This script (*test/models/createmodel_AplusB.py*) provides a minimal example of how to save the model, which also automatically detect your tensorflow version to save the corresponding format.
```sh
# Create *.pb format A + B model:
python3 test/models/createmodel_AplusB.py PB

# Create SavedModel format A + B model:
python3 test/models/createmodel_AplusB.py SavedModel
```

For SavedModel format, you might need the tensorflow *saved_model_cli* tools to check the model's *input*, *output* and *tags*, the document can be found [here](https://www.tensorflow.org/guide/saved_model). 

### C++ core predictor
A minimal example of loading A + B model can be found at *test/test_AplusB.cpp*

Load the model:
```c++
// Load the PB graph:
Predictor pd("graph.pb");

// Load the SavedModel format, please specify the model dir and tag name
Predictor pd_s("model_dir", "your_tags");
```

Print information of nodes in the model:
```c++
// Print all nodes name:
pd.print_operations();

// Print information of specific node:
pd.print_operations("your_operation_name");
```

Register the nodes for input and output:
```c++
// Please substitute the "your_input_name" and "your_output_name" by your real node name
pd.regist_node("your_input_name", Predictor::INPUT_NODE);
pd.regist_node("your_output_name", Predictor::OUTPUT_NODE);
```

If your model is with unknown dimension (normally the number of examples you want to predict, for example the shape is `[-1, 256, 256]`), set the unknown dimension with exact integer before feed data to the node:
```c++
pd.set_data_count(n_data);  // n_data is int type 
```

This project provides several features for feeding/extracting data to the tensor.
- The data will be automatically cast to the data type defined by tensorflow model.
- The data can be transposed while being fed or extracted. 
- Eigen methods are enabled to support a vectorized type casting and transpose 

Full usage:

```c++
Predictor::set_node_data(std::string node_name, T* data, int data_size, Predictor::DataLayout, Predictor::CopyMethod)
```

- `T` is the raw data pointer to your data, which can be: 
`float double int32_t int16_t int8_t uint8_t uint16_t uint32_t`
- `data_size` is the number of data from `T*` that you want to feed
- `Predictor::Datalayout` can be `Predictor::RowMajor` or `Predictor::ColumnMajor`, to determine whether transpose is needed
- `Predictor::CopyMethod` can be `Predictor::Simple` or `Predictor::Eigen`, to determine whether Eigen vectorization is enabled.

There is also some simplified overload with `std::vector`, for example:
```c++
std::vector<float> data_in = {/*Some your data*/}
pd.set_node_data("your_input_name", data_in); //Default for RowMajor and Eigen
```

Run model:
```c++
pd.run(); //Default for RowMajor and Eigen
```

Extract data (the same usage with feeding data):
```c++
int n_data_volume // This need to be the same with the total data number of your output data, for example the output node is shape: [3,3,2], then n_data_volume should be 3 * 3 * 2 = 18 
std::vector data_out(n_data_volume)
pd.get_node_data("your_output_name", data_out); //Default for RowMajor and Eigen
```



### Fortran API usage:
This predictor also provides two extensions to Fortran, for the convenience of physicists or engineers who want to integrate AI techniques into PDE simulation codes. A minimal test program can be found in *test/fortran_test_AplutB.f90* . The `iso_c_binding` module is required to provide the size of a c-pointer according to your compiler, and also the `predictor` module should be used to provide Fortran API for the program. 

Therefore the program should at least contain:
```fortran
program main
    ! use TF_predictor interface
    use predictor
    ! Supply the size of C-pointer
    use iso_c_binding, only: C_ptr

    ! Some of your code
    ! ...

end program main
```

Create the predictor:
```fortran
! Type declaration:
TYPE(C_ptr) :: pd

! Create the predictor from *.pb
pd = C_createPredictor("your_pb_file.pb")
! Create the predictor from SavedModel format
pd_s = C_createPredictor("your_model_dir", "your_tag")
```

Register input and output:
```fortran
! Register input node
call C_PredictorRegisterInputNode(pd, "your_input_node")
! Register output node
call C_PredictorRegisterOutputNode(pd, "your_output_node")
```

Feed data to input node:
```fortran
! Data type declaration, support real4 real8 integer4, up to 6d array:
REAL(kind=8), dimension(5) :: input_1d_double
REAL(kind=4), dimension(2,3) :: input_2d_float
INTEGER(kind=4), dimension(2,2,2) :: input_3d_int

! Feed data to input nodes:
call C_PredictorSetNodeData(pd, "your_input_node", input_1d_double, size(input_1d_double))
call C_PredictorSetNodeData(pd, "your_input_node_2", input_2d_float, size(input_2d_float))
call C_PredictorSetNodeData(pd, "your_input_node_3", input_3d_int, size(input_3d_int))
```
Run the model:
```fortran
call C_PredictorRun(pd)
```

Extract data from output node (same as feeding data to input):

```fortran
call C_PredictorSetNodeData(pd, "your_output_node", your_output_array, size(your_output_array))
```

Finally, please remember to delete the predictor in case of memory leakage:
 ```fortran
call C_deletePredictor(pd)
```

