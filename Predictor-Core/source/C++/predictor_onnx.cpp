#include "predictor.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <cstring>
#include <memory>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <complex>


#include <unsupported/Eigen/CXX11/Tensor>

#ifdef _DEBUG
#define DEBUG_EXECUTE(x) (x)
#else
#define DEBUG_EXECUTE(x) do {} while (0)
#endif

// #if (defined EIGEN_DEFAULT_DENSE_INDEX_TYPE) 
//   #undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
// #endif

// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE size_t

#define EIGEN_DONT_PARALLELIZE
// typedef enum ONNXTensorElementDataType {
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   // maps to c type float
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,   // maps to c type uint8_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,    // maps to c type int8_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,  // maps to c type uint16_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,   // maps to c type int16_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,   // maps to c type int32_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // maps to c type int64_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,  // maps to c++ type std::string
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,      // maps to c type double
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,      // maps to c type uint32_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,      // maps to c type uint64_t
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,   // complex with float32 real and imaginary components
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,  // complex with float64 real and imaginary components
//   ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16     // Non-IEEE floating-point format based on IEEE754 single-precision
// } ONNXTensorElementDataType;


std::map<ONNXTensorElementDataType, std::string> DT_TO_STRING = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED          , "UNDEFINED"     },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT              , "FLOAT"         },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8              , "UINT8"         },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8               , "INT8"          },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16             , "UINT16"        },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16              , "INT16"         },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32              , "INT32"         },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64              , "INT64"         },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING             , "STRING"        },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL               , "BOOL"          },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16            , "FLOAT16"       },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE             , "DOUBLE"        },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32             , "UINT32"        },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64             , "UINT64"        },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64          , "COMPLEX64"     },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128         , "COMPLEX128"    },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16           , "BFLOAT16"      }
};

template <typename T>
ONNXTensorElementDataType deduce_type() {
    if (std::is_same<T, float>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (std::is_same<T, double>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    if (std::is_same<T, int32_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    if (std::is_same<T, uint8_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    if (std::is_same<T, int16_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    if (std::is_same<T, int8_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    if (std::is_same<T, int64_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    if (std::is_same<T, uint16_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    if (std::is_same<T, uint32_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    if (std::is_same<T, uint64_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    if (std::is_same<T, bool>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    if (std::is_same<T, std::string>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    if (std::is_same<T, std::complex<float> >::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    if (std::is_same<T, std::complex<double> >::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    if (std::is_same<T, Ort::Float16_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (std::is_same<T, Ort::BFloat16_t>::value)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    

    std::cerr << "Currently not supported data type." << std::endl;
    return static_cast<ONNXTensorElementDataType>(-100);
}


// instantiate template functions
template ONNXTensorElementDataType deduce_type<float>();
template ONNXTensorElementDataType deduce_type<double>();
template ONNXTensorElementDataType deduce_type<int8_t>();
template ONNXTensorElementDataType deduce_type<int16_t>();
template ONNXTensorElementDataType deduce_type<int32_t>();
template ONNXTensorElementDataType deduce_type<int64_t>();
template ONNXTensorElementDataType deduce_type<uint8_t>();
template ONNXTensorElementDataType deduce_type<uint16_t>();
template ONNXTensorElementDataType deduce_type<uint32_t>();
template ONNXTensorElementDataType deduce_type<uint64_t>();
template ONNXTensorElementDataType deduce_type<bool>();
template ONNXTensorElementDataType deduce_type<std::string>();
template ONNXTensorElementDataType deduce_type<std::complex<float>>();
template ONNXTensorElementDataType deduce_type<std::complex<double>>();
template ONNXTensorElementDataType deduce_type<Ort::Float16_t>();
template ONNXTensorElementDataType deduce_type<Ort::BFloat16_t>();



// print list
void print_shape(std::vector<int64_t> shape) {
    if (shape.empty()) {
        std::cout << "empty shape" << std::endl;
        return;
    }
    std::cout << "[ ";
    for (size_t i = 0; i < shape.size()-1; i++) {
        std::cout << shape[i] << ", ";
    }
    std::cout << shape[shape.size()-1];
    std::cout << " ]" << std::endl;
}


// multiplication of each elements
int64_t get_numelements_from_shape(std::vector<int64_t> shape) {
    if (shape.empty()) {
        return 0;
    }

    int64_t num_ele = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] <= 0) {
            std::cout << "Rank: " << i << " in shape is unknown, cannot obtain total length." << std::endl;
            return -1;
        }
        num_ele *= shape[i];
    }
    return num_ele;
}

struct NodeInfo {
    std::string                    name;
    std::vector<int64_t>           shape;
    ONNXTensorElementDataType      type;
    NodeInfo()
        : type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED){};
};


struct ONNXModel {
    Ort::Env*                          env;
    Ort::SessionOptions*               options;
    Ort::Session*                      session;
    Ort::AllocatorWithDefaultOptions*  allocator;
    Ort::RunOptions*                   runoptions;
    // Ort::MemoryInfo*                   memoryinfo;
    ONNXModel()
        : env(nullptr), options(nullptr), session(nullptr), allocator(nullptr), runoptions(nullptr) {};
        // ,memoryinfo(nullptr)
};

// Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)


class PredictorImpl {
public:
    int64_t                    data_count;

    size_t                     input_count;
    size_t                     output_count;

    std::vector<NodeInfo>      input_nodes;
    std::vector<NodeInfo>      output_nodes;

    std::vector<Ort::Value>    input_tensors;
    std::vector<Ort::Value>    output_tensors;

    std::vector<char*>         input_names;
    std::vector<char*>         output_names;

    ONNXModel model;
};




//---------------------------------------Constructor-----------------------------------------------

#ifdef _CPU_SERIAL
Predictor::Predictor(std::string pbfile) : Predictor(pbfile, 1, 1) {}
Predictor::Predictor(std::string folder, std::string tag) : Predictor(folder, tag, 1, 1) {}
#else
Predictor::Predictor(std::string pbfile) : Predictor(pbfile, 0, 0) {}
Predictor::Predictor(std::string folder, std::string tag) : Predictor(folder, tag, 0, 0) {}
#endif



Predictor::Predictor(std::string pbfile, int intra_op_parallelism_threads, int inter_op_parallelism_threads)
    : d(nullptr) {
    d = new PredictorImpl();
    d->data_count = -1;

    std::cout << "Initializing model from onnx graph model (*.pb)..." << std::endl;
    std::cout << "Initializing ONNX model: " << pbfile  << "  ..." << std::endl;
    d->model.env = new Ort::Env();
    d->model.options = new Ort::SessionOptions();
    d->model.options->SetIntraOpNumThreads(intra_op_parallelism_threads);
    d->model.options->SetInterOpNumThreads(inter_op_parallelism_threads);
    d->model.session = new Ort::Session(*(d->model.env), pbfile.c_str(), *(d->model.options));
    d->model.allocator = new Ort::AllocatorWithDefaultOptions();
    d->model.runoptions = new Ort::RunOptions();
    // d->model.memoryinfo = new Ort::MemoryInfo::CeateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::cout << "Initialization Done!" << std::endl;

    // Auto Register model I/O
    std::cout << "Parsing model's input/output nodes ..." << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    d->input_count = d->model.session->GetInputCount();
    std::cout << "Input count:"  << d->input_count << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    d->input_names.reserve(d->input_count);

    for (size_t i = 0; i < d->input_count; i++) {
        NodeInfo temp_info;
        Ort::AllocatedStringPtr temp_name = d->model.session->GetInputNameAllocated(i, *(d->model.allocator));
        temp_info.name = temp_name.get();
        d->input_names[i] = new char[1024];
        std::strcpy(d->input_names[i], temp_name.get());
        std::cout << "Name: " << d->input_names[i] << "\n";
        Ort::TypeInfo input_info = d->model.session->GetInputTypeInfo(i);
        Ort::Unowned< Ort::TensorTypeAndShapeInfo > input_type_and_shape_info = input_info.GetTensorTypeAndShapeInfo();
        temp_info.type = input_type_and_shape_info.GetElementType();
        std::cout << "Datatype: " << DT_TO_STRING[temp_info.type] << "\n";
        size_t 	dim_count = input_type_and_shape_info.GetDimensionsCount();
        temp_info.shape = input_type_and_shape_info.GetShape();
        std::cout << "Shape: " ;
        print_shape(temp_info.shape);
        d->input_nodes.push_back(temp_info);
        std::cout << "--------------------------------------" << std::endl;
    }
    assert(d->input_nodes.size() == d->input_count);


    std::cout << "--------------------------------------" << std::endl;
    d->output_count = d->model.session->GetOutputCount();
    std::cout << "Output count:"  << d->output_count << std::endl;
    std::cout << "--------------------------------------" << std::endl;


    d->output_names.reserve(d->output_count);

    for (size_t i = 0; i < d->output_count; i++) {
        NodeInfo temp_info;
        Ort::AllocatedStringPtr temp_name = d->model.session->GetOutputNameAllocated(i, *(d->model.allocator));
        temp_info.name = temp_name.get();
        d->output_names[i] = new char[1024];
        std::strcpy(d->output_names[i], temp_name.get());
        std::cout << "Name: " << d->output_names[i] << "\n";
        Ort::TypeInfo output_info = d->model.session->GetOutputTypeInfo(i);
        Ort::Unowned< Ort::TensorTypeAndShapeInfo > output_type_and_shape_info = output_info.GetTensorTypeAndShapeInfo();
        temp_info.type = output_type_and_shape_info.GetElementType();
        std::cout << "Datatype: " << DT_TO_STRING[temp_info.type] << "\n";
        size_t 	dim_count = output_type_and_shape_info.GetDimensionsCount();
        temp_info.shape = output_type_and_shape_info.GetShape();
        std::cout << "Shape: " ;
        print_shape(temp_info.shape);
        d->output_nodes.push_back(temp_info);
        std::cout << "--------------------------------------" << std::endl;
    }
    

    assert(d->output_nodes.size() == d->output_count);

}

Predictor::Predictor(std::string folder, std::string tags, int intra_op_parallelism_threads, int inter_op_parallelism_threads)
    : Predictor(folder, intra_op_parallelism_threads, inter_op_parallelism_threads) {
    
    std::cerr << "Warning: onnx runtime only need one [model_path] without [tags]" << std::endl;
    std::cerr << "Using: [" << folder << "] as model path" << std::endl;


}

//----------------------------------------------Destructor--------------------------------------

Predictor::~Predictor() {

    for (size_t i = 0; i < d->input_nodes.size(); i++) {
        if (nullptr != d->input_names[i]) {
            delete d->input_names[i];
        }
    }

    for (size_t i = 0; i < d->output_nodes.size(); i++) {
        if (nullptr != d->output_names[i]) {
            delete d->output_names[i];
        }
    }

    if (nullptr != d->model.env) {
        delete d->model.env;
    }
    if (nullptr != d->model.options) {
        delete d->model.options;
    }
    if (nullptr != d->model.session) {
        delete d->model.session;
    }
    if (nullptr != d->model.allocator) {
        delete d->model.allocator;
    }
    // if (nullptr != d->model.memoryinfo) {
    //     delete d->model.memoryinfo;
    // }

    delete d;
}

void Predictor::print_operations(std::string node_name) {
    if (d->model.session == nullptr) {
        std::cerr << "Model is not properly initialized, please check your initialization." << std::endl;
        return;
    }

    if (d->input_nodes.empty()) {
        std::cerr << "Warning: No input node recorded in the model." << std::endl;
    }
    
    if (d->output_nodes.empty()) {
        std::cerr << "Warning: No output node recorded in the model." << std::endl;
    }

    if ((d->output_nodes.empty()) && (d->input_nodes.empty())) {
        std::cerr << "Error: No input/output node recorded in the model." << std::endl;
        return;
    }

    for (size_t i = 0; i < d->input_nodes.size(); i++) {
        if (d->input_nodes[i].name == node_name) {
            std::cout << "--------------------------------------" << std::endl;
            std::cout << "Input node: " << d->input_nodes[i].name << "\n";
            std::cout << "Index of input: " << i << "\n";
            std::cout << "Data type: " << DT_TO_STRING[d->input_nodes[i].type] << "\n";
            std::cout << "Data shape: " << "\n";
            print_shape(d->input_nodes[i].shape);
            return;
        }
    }

    for (size_t i = 0; i < d->output_nodes.size(); i++) {
        if (d->output_nodes[i].name == node_name) {
            std::cout << "--------------------------------------" << std::endl;
            std::cout << "Output node: " << d->output_nodes[i].name << "\n";
            std::cout << "Index of output: " << i << "\n";
            std::cout << "Data type: " << DT_TO_STRING[d->output_nodes[i].type] << "\n";
            std::cout << "Data shape: " << "\n";
            print_shape(d->output_nodes[i].shape);
            return;
        }
    }

    std::cerr << "Warning: No node named \"" << node_name << "\" in the input/output field." <<std::endl;

}



void Predictor::print_operations() {
    if (d->model.session == nullptr) {
        std::cerr << "Model is not properly initialized, please check your initialization." << std::endl;
        return;
    }

    if (d->input_nodes.empty()) {
        std::cerr << "Warning: No input node recorded in the model." << std::endl;
    }
    
    if (d->output_nodes.empty()) {
        std::cerr << "Warning: No output node recorded in the model." << std::endl;
    }

    if ((d->output_nodes.empty()) && (d->input_nodes.empty())) {
        std::cerr << "Error: No input/output node recorded in the model." << std::endl;
        return;
    }

    for (size_t i = 0; i < d->input_nodes.size(); i++) {
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Input node: " << d->input_nodes[i].name << "\n";
        std::cout << "Index of input: " << i << "\n";
        std::cout << "Data type: " << DT_TO_STRING[d->input_nodes[i].type] << "\n";
        std::cout << "Data shape: " << "\n";
        print_shape(d->input_nodes[i].shape);
    }

    for (size_t i = 0; i < d->output_nodes.size(); i++) {
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Output node: " << d->output_nodes[i].name << "\n";
        std::cout << "Index of output: " << i << "\n";
        std::cout << "Data type: " << DT_TO_STRING[d->output_nodes[i].type] << "\n";
        std::cout << "Data shape: " << "\n";
        print_shape(d->output_nodes[i].shape);
    }

}

void Predictor::regist_node(std::string name, Predictor::NodeType type) {
    DEBUG_EXECUTE(std::cerr << "Warning: In Onnx backends, the I/O nodes are auto-registered." << std::endl);
#ifdef _DEBUG
    std::cout << "Checking whether the register infomation matches:" << std::endl;
    if (type == Predictor::INPUT_NODE) {
        std::cout << "Node registered name: " << name << ", registered as INPUT." << std::endl;
    } else {
        std::cout << "Node registered name: " << name << ", registered as OUTPUT." << std::endl;
    }
    std::cout << "The actual information in session graph is: \n";
    print_operations(name);
# endif
}

void Predictor::set_data_count(int64_t cnt) {
    assert(cnt > 0);

    if (d->data_count == cnt) {
        DEBUG_EXECUTE(std::cerr << "Same length of data instances, tensors no need to reset." << std::endl);
        return;
    } else if (-1 != d->data_count) {
        DEBUG_EXECUTE(std::cerr << "warning: overwriting data count" << std::endl);
        DEBUG_EXECUTE(std::cerr << "warning: Changing size of containers" << std::endl);

        d->input_tensors.clear();
        d->output_tensors.clear();
    }
    

    for (size_t i = 0; i < d->input_count; i++) {
        std::vector<int64_t> temp_shape = d->input_nodes[i].shape;
        for (size_t i = 0; i < temp_shape.size(); i++) {
            if (temp_shape[i] == -1) {
                temp_shape[i] = cnt;
            }
        }
        d->input_tensors.push_back(Ort::Value::CreateTensor(*(d->model.allocator), temp_shape.data(), temp_shape.size(), d->input_nodes[i].type));
    }

    for (size_t i = 0; i < d->output_count; i++) {
        std::vector<int64_t> temp_shape = d->output_nodes[i].shape;
        for (size_t i = 0; i < temp_shape.size(); i++) {
            if (temp_shape[i] == -1) {
                temp_shape[i] = cnt;
            }
        }
        d->output_tensors.push_back(Ort::Value::CreateTensor(*(d->model.allocator), temp_shape.data(), temp_shape.size(), d->output_nodes[i].type));
    }
    d->data_count = cnt;

    assert(d->input_tensors.size() == d->input_count);
    assert(d->output_tensors.size() == d->output_count);

}

int64_t Predictor::get_data_count() {
    return d->data_count;
}

//  Initialize finished

//-------------------------------------------------------------------------------------------------
template <typename T_data, typename T_tensor>
static void set_tensor_data_row_simple(T_data* src, Ort::Value* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_row_simple" << std::endl);
    T_tensor* buff_tensor = dst->GetTensorMutableData<T_tensor>();
    size_t nelems = dst->GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < nelems; ++i) {
        buff_tensor[i] = src[i];
    }
}

/// @brief partial transpose dst[i,j,k, :] = src[:, k,j,i].T
/// @param rank         number of dimensions
/// @param slice_rank   number of fixed dimensions (known index)
template <typename T, typename U>
static void transpose_slice(size_t rank, size_t slice_rank, const int64_t dst_shape[],
                            T* dst, const int64_t dst_step[], size_t dst_offset,
                            const U* src, const int64_t src_step[], size_t src_offset) {
    size_t limit = dst_shape[slice_rank];
    size_t dstep = dst_step[slice_rank];
    size_t sstep = src_step[rank - slice_rank - 1];

    if (slice_rank == rank - 1) {
        for (size_t i = 0; i < limit; ++i) {
            dst[dst_offset] = src[src_offset];
            dst_offset += dstep;
            src_offset += sstep;
        }
        return;
    }

    for (size_t i = 0; i < limit; ++i) {
        transpose_slice(rank, slice_rank + 1, dst_shape,
            dst, dst_step, dst_offset,
            src, src_step, src_offset);
        dst_offset += dstep;
        src_offset += sstep;
    }
}

template <typename T, typename U>
static void simple_transpose(const U* src, size_t rank, const int64_t dims[], T* dst) {

    // get tensor shape
    const int64_t* src_shape = dims;
    int64_t* dst_shape = new int64_t[rank];
    for (size_t i = 0; i < rank; ++i) {
        dst_shape[i] = src_shape[rank - i - 1];
    }

    // get step of each dimension
    int64_t *dst_step = new int64_t[rank];
    int64_t *src_step = new int64_t[rank];
    dst_step[rank - 1] = 1;
    src_step[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        dst_step[i] = dst_step[i + 1] * dst_shape[i + 1];
        src_step[i] = src_step[i + 1] * src_shape[i + 1];
    }

    transpose_slice<T, U>(rank, 0, dst_shape,
        dst, dst_step, 0,
        src, src_step, 0);

    delete[] dst_shape;
    delete[] dst_step;
    delete[] src_step;
}


// /*
// ndarray dim [d4, d3, d2, d1] = [5,6,7,8]
// slice [a, :, c, d]
// offset, step

// step(2) = [d3*d2*d1, d2*d1, d1, 1]
// [a,b,c,d] -- [a,b,c+1, d]

// flatten (row-major)
// [0,0,0,0]~[0,0,0,7]
// [0,0,1,0]~[0,0,1,7]

// dst = src.T (N)

// dst[0, :] = src[:, 0].T (N-1)
// dst[1, :] = src[:, 1].T
// dst[2, :] = src[:, 2].T
// dst[3, :] = src[:, 3].T

// */

// multidimension transpose set
// dst[i,j,k,...] = src[...,k,j,i]
template <typename T_data, typename T_tensor>
static void set_tensor_data_col_simple(T_data* src, Ort::Value* dst) {
    T_tensor* buff_tensor = dst->GetTensorMutableData<T_tensor>();

    size_t rank = dst->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    if (1 == rank) {
        set_tensor_data_row_simple<T_data, T_tensor>(src, dst);
        return;
    }
    int64_t* dst_shape = new int64_t[rank];
    
    dst->GetTensorTypeAndShapeInfo().GetDimensions(dst_shape, rank);
    int64_t* src_shape = new int64_t[rank];
    for (int i = 0; i < rank; ++i) {
        src_shape[i] = dst_shape[rank - i - 1];
    }

    simple_transpose<T_tensor, T_data>(src, rank, src_shape, buff_tensor);
    delete[] src_shape;
    delete[] dst_shape;
}

template <typename T_data, typename T_tensor>
static void set_tensor_data_row_eigen(T_data* src, Ort::Value* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_row_eigen" << std::endl);
    int64_t total_length = dst->GetTensorTypeAndShapeInfo().GetElementCount();
    Eigen::array<Eigen::Index, 1> shape_arr({total_length});
    Eigen::TensorMap<Eigen::Tensor<T_data, 1, Eigen::RowMajor>> data_wrapper(src, shape_arr);
    T_tensor* buff_tensor = dst->GetTensorMutableData<T_tensor>();
    Eigen::TensorMap<Eigen::Tensor<T_tensor, 1, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_arr);
    tensor_wrapper = data_wrapper.template cast<T_tensor>();
}

template <typename T_data, typename T_tensor>
static void set_tensor_data_col_eigen(T_data* src, Ort::Value* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_col_eigen" << std::endl);
    size_t rank = dst->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    T_tensor* buff_tensor = dst->GetTensorMutableData<T_tensor>();
    int64_t* dst_shape = new int64_t[rank];
    dst->GetTensorTypeAndShapeInfo().GetDimensions(dst_shape, rank);

    switch (rank) {
        case 1: {
            set_tensor_data_row_eigen<T_data, T_tensor>(src, dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({dst_shape[0],
                                                        dst_shape[1]});
            Eigen::TensorMap<Eigen::Tensor<T_tensor, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);
            Eigen::array<int, 2> shuffling({1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({dst_shape[4],
                                                      dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3],
                                                        dst_shape[4]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({dst_shape[5],
                                                      dst_shape[4],
                                                      dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3],
                                                        dst_shape[4],
                                                        dst_shape[5]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 6, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 6> shuffling({5, 4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        default: {
            std::cout << "Unsupported dimensions, set data failed." << std::endl;
            return;
        } break;
    }
    delete[] dst_shape;
}

template <typename T_data>
static void set_tensor_data_col_eigen_sametype(T_data* src, Ort::Value* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_col_eigen_sametype" << std::endl);
    size_t rank = dst->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    T_data* buff_tensor = dst->GetTensorMutableData<T_data>();
    int64_t* dst_shape = new int64_t[rank];
    dst->GetTensorTypeAndShapeInfo().GetDimensions(dst_shape, rank);

    switch (rank) {
        case 1: {
            std::copy_n(src, dst->GetTensorTypeAndShapeInfo().GetElementCount(), buff_tensor);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({dst_shape[0],
                                                        dst_shape[1]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 2> shuffling({1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(src, shape_data);


            Eigen::array<Eigen::Index, 3> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({dst_shape[4],
                                                      dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3],
                                                        dst_shape[4]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({dst_shape[5],
                                                      dst_shape[4],
                                                      dst_shape[3],
                                                      dst_shape[2],
                                                      dst_shape[1],
                                                      dst_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({dst_shape[0],
                                                        dst_shape[1],
                                                        dst_shape[2],
                                                        dst_shape[3],
                                                        dst_shape[4],
                                                        dst_shape[5]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 6> shuffling({5, 4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        default: {
            std::cout << "Unsupported dimensions, set data failed." << std::endl;
            return;
        } break;
    }
    delete[] dst_shape;
}

template <typename T>
bool Predictor::set_node_data(std::string name, T* p_data, int array_size) {
    return this->set_node_data(name, p_data, array_size, Predictor::RowMajor, Predictor::Eigen);
}

template bool Predictor::set_node_data<float>(std::string name, float* p_data, int array_size);
template bool Predictor::set_node_data<double>(std::string name, double* p_data, int array_size);
template bool Predictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size);
template bool Predictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size);
template bool Predictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size);
template bool Predictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size);
template bool Predictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size);
template bool Predictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size);

template <typename T>
bool Predictor::set_node_data(std::string name, T* p_data, int array_size, Predictor::DataLayout layout) {
    return this->set_node_data(name, p_data, array_size, layout, Predictor::Eigen);
}

template bool Predictor::set_node_data<float>(std::string name, float* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<double>(std::string name, double* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Predictor::DataLayout layout);

template <typename T>
bool Predictor::set_node_data(std::string name, T* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method) {
    
    size_t tensor_index = -1;
    for (size_t i = 0; i < d->input_count; i++) {
        if (d->input_nodes[i].name == name) {
            tensor_index = i;
        }
    }

    if (tensor_index == -1) {
        std::cerr << "fatal: input node " << name << " not found!" << std::endl;
        return false;
    }

    if (d->input_tensors.size() == 0) {
        std::cerr << "fatal: input tensors not created, please set data count first!" << std::endl;
        return false;
    }

    
    std::vector<int64_t> shape_information = d->input_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType type_information = d->input_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementType();
    assert(0 != shape_information.size());

    size_t total_element = d->input_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementCount();
    if (total_element == array_size) {
        DEBUG_EXECUTE(std::cout << "Total number of data checked same " << std::endl);
    } else {
        std::cout << "Total number of data checked incompatible, [Inner: " << total_element << "] vs [Outer: " 
        << array_size <<"]" << std::endl;
        return false;
    }

    if (deduce_type<T>() == type_information) {
        switch (layout) {
            case Predictor::RowMajor: {
                T* buff_in = d->input_tensors[tensor_index].GetTensorMutableData<T>();
                std::copy_n(p_data, d->input_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementCount(), buff_in);
                return true;
            } break;
            case Predictor::ColumnMajor: {
                set_tensor_data_col_eigen_sametype<T>(p_data, &(d->input_tensors[tensor_index]));
                return true;
            } break;

            default: {
                DEBUG_EXECUTE(std::cout << "Different datatype between tensor definition and data source, auto casting enabled." << std::endl);
            } break;
        }
    }

    switch (method) {
        case Predictor::Eigen: {
            switch (layout) {
                case Predictor::RowMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            set_tensor_data_row_eigen<T, float>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            set_tensor_data_row_eigen<T, double>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            set_tensor_data_row_eigen<T, int32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            set_tensor_data_row_eigen<T, int16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            set_tensor_data_row_eigen<T, int8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            set_tensor_data_row_eigen<T, uint32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            set_tensor_data_row_eigen<T, uint16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            set_tensor_data_row_eigen<T, uint8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            set_tensor_data_col_eigen<T, float>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            set_tensor_data_col_eigen<T, double>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            set_tensor_data_col_eigen<T, int32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            set_tensor_data_col_eigen<T, int16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            set_tensor_data_col_eigen<T, int8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            set_tensor_data_col_eigen<T, uint32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            set_tensor_data_col_eigen<T, uint16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            set_tensor_data_col_eigen<T, uint8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                default: {
                    std::cout << "fatal: Unsupported layout, set data didn't succeed." << std::endl;
                    return false;
                } break;
            }
        } break;

        case Predictor::Simple: {
            switch (layout) {
                case Predictor::RowMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            set_tensor_data_row_simple<T, float>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            set_tensor_data_row_simple<T, double>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            set_tensor_data_row_simple<T, int32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            set_tensor_data_row_simple<T, int16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            set_tensor_data_row_simple<T, int8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            set_tensor_data_row_simple<T, uint32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            set_tensor_data_row_simple<T, uint16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            set_tensor_data_row_simple<T, uint8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            set_tensor_data_col_simple<T, float>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            set_tensor_data_col_simple<T, double>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            set_tensor_data_col_simple<T, int32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            set_tensor_data_col_simple<T, int16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            set_tensor_data_col_simple<T, int8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            set_tensor_data_col_simple<T, uint32_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            set_tensor_data_col_simple<T, uint16_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            set_tensor_data_col_simple<T, uint8_t>(p_data, &(d->input_tensors[tensor_index]));
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                default: {
                    std::cout << "fatal: Unsupported layout, set data didn't succeed." << std::endl;
                    return false;
                } break;
            }
        } break;

        default: {
            std::cout << "fatal: Unsupported method, set data didn't succeed." << std::endl;
            return false;
        } break;
    }
    return true;
}


template bool Predictor::set_node_data<float>(std::string name, float* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<double>(std::string name, double* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
//-------------------------------------------------------------------------------------------------
// std::vector wrapper
template <typename T>
bool Predictor::set_node_data(std::string name, std::vector<T>& data) {
    return this->set_node_data(name, data.data(), data.size(), Predictor::RowMajor, Predictor::Eigen);
}

template bool Predictor::set_node_data<float>(std::string name, std::vector<float>& data);
template bool Predictor::set_node_data<double>(std::string name, std::vector<double>& data);
template bool Predictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data);
template bool Predictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data);
template bool Predictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data);
template bool Predictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data);
template bool Predictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data);
template bool Predictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data);


template <typename T>
bool Predictor::set_node_data(std::string name, std::vector<T>& data, Predictor::DataLayout layout) {
    return this->set_node_data(name, data.data(), data.size(), layout, Predictor::Eigen);
}

template bool Predictor::set_node_data<float>(std::string name, std::vector<float>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<double>(std::string name, std::vector<double>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Predictor::DataLayout layout);
template bool Predictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Predictor::DataLayout layout);

template <typename T>
bool Predictor::set_node_data(std::string name, std::vector<T>& data, Predictor::DataLayout layout, Predictor::CopyMethod method) {
    return this->set_node_data(name, data.data(), data.size(), layout, method);
}

template bool Predictor::set_node_data<float>(std::string name, std::vector<float>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<double>(std::string name, std::vector<double>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);


//-------------------------------------------------------------------------------------------------
//Get data function implementation

template <typename T_tensor, typename T_data>
static void get_tensor_data_row_simple(Ort::Value* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_row_simple" << std::endl);
    T_tensor* buff_tensor = src->GetTensorMutableData<T_tensor>();
    size_t nelems = src->GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < nelems; ++i) {
        dst[i] = buff_tensor[i];
    }
}

template <typename T_tensor, typename T_data>
static void get_tensor_data_col_simple(Ort::Value* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_col_simple" << std::endl);
    T_tensor* buff_tensor = src->GetTensorMutableData<T_tensor>();
    size_t rank = src->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    if (1 == rank) {
        get_tensor_data_row_simple<T_tensor, T_data>(src, dst);
        return;
    }
    // get tensor shape
    int64_t* src_shape = new int64_t[rank];
    src->GetTensorTypeAndShapeInfo().GetDimensions(src_shape, rank);
    simple_transpose<T_data, T_tensor>(buff_tensor, rank, src_shape, dst);
    delete[] src_shape;

}


template <typename T_tensor, typename T_data>
static void get_tensor_data_row_eigen(Ort::Value* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_row_eigen" << std::endl);
    int64_t total_length = src->GetTensorTypeAndShapeInfo().GetElementCount();
    Eigen::array<Eigen::Index, 1> shape_arr({total_length});
    Eigen::TensorMap<Eigen::Tensor<T_data, 1, Eigen::RowMajor>> data_wrapper(dst, shape_arr);
    T_tensor* buff_tensor = src->GetTensorMutableData<T_tensor>();
    Eigen::TensorMap<Eigen::Tensor<T_tensor, 1, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_arr);
    data_wrapper = tensor_wrapper.template cast<T_data>();
}

template <typename T_tensor, typename T_data>
static void get_tensor_data_col_eigen(Ort::Value* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_col_eigen" << std::endl);
    size_t rank = src->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    T_tensor* buff_tensor = src->GetTensorMutableData<T_tensor>();
    int64_t* src_shape = new int64_t[rank];
    src->GetTensorTypeAndShapeInfo().GetDimensions(src_shape, rank);

    switch (rank) {
        case 1: {
            get_tensor_data_row_eigen<T_tensor, T_data>(src, dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({src_shape[0],
                                                        src_shape[1]});
            Eigen::TensorMap<Eigen::Tensor<T_tensor, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);
            Eigen::array<int, 2> shuffling({1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({src_shape[4],
                                                      src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3],
                                                        src_shape[4]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({src_shape[5],
                                                      src_shape[4],
                                                      src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3],
                                                        src_shape[4],
                                                        src_shape[5]});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 6, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 6> shuffling({5, 4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        default: {
            std::cout << "Unsupported dimensions, get data failed." << std::endl;
            return;
        } break;
    }
    delete[] src_shape;
}

template <typename T_data>
static void get_tensor_data_col_eigen_sametype(Ort::Value* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_col_eigen_sametype" << std::endl);
    size_t rank = src->GetTensorTypeAndShapeInfo().GetDimensionsCount();
    T_data* buff_tensor = src->GetTensorMutableData<T_data>();
    int64_t* src_shape = new int64_t[rank];
    src->GetTensorTypeAndShapeInfo().GetDimensions(src_shape, rank);

    switch (rank) {
        case 1: {
            std::copy_n(buff_tensor, src->GetTensorTypeAndShapeInfo().GetElementCount(), dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({src_shape[0],
                                                        src_shape[1]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 2> shuffling({1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({src_shape[4],
                                                      src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3],
                                                        src_shape[4]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({src_shape[5],
                                                      src_shape[4],
                                                      src_shape[3],
                                                      src_shape[2],
                                                      src_shape[1],
                                                      src_shape[0]});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({src_shape[0],
                                                        src_shape[1],
                                                        src_shape[2],
                                                        src_shape[3],
                                                        src_shape[4],
                                                        src_shape[5]});

            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 6> shuffling({5, 4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        default: {
            std::cout << "Unsupported dimensions, get data failed." << std::endl;
            return;
        } break;
    }
    delete[] src_shape;
}

template <typename T>
bool Predictor::get_node_data(std::string name, T* p_data, int array_size) {
    return this->get_node_data(name, p_data, array_size, Predictor::RowMajor, Predictor::Eigen);
}

template bool Predictor::get_node_data<float>(std::string name, float* p_data, int array_size);
template bool Predictor::get_node_data<double>(std::string name, double* p_data, int array_size);
template bool Predictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size);
template bool Predictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size);
template bool Predictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size);
template bool Predictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size);
template bool Predictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size);
template bool Predictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size);

template <typename T>
bool Predictor::get_node_data(std::string name, T* p_data, int array_size, Predictor::DataLayout layout) {
    return this->get_node_data(name, p_data, array_size, layout, Predictor::Eigen);
}

template bool Predictor::get_node_data<float>(std::string name, float* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<double>(std::string name, double* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Predictor::DataLayout layout);
// template bool Predictor::get_node_data<uint64_t>(std::string name, std::vector<uint64_t>& data, Predictor::DataLayout layout);
// template bool Predictor::get_node_data<int64_t>(std::string name, std::vector<int64_t>& data, Predictor::DataLayout layout);

template <typename T>
bool Predictor::get_node_data(std::string name, T* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method) {
    size_t tensor_index = -1;
    for (size_t i = 0; i < d->output_count; i++) {
        if (d->output_nodes[i].name == name) {
            tensor_index = i;
        }
    }

    if (tensor_index == -1) {
        std::cerr << "fatal: output node " << name << " not found!" << std::endl;
        return false;
    }

    if (d->output_tensors.size() == 0) {
        std::cerr << "fatal: output tensors not created, please set data count first!" << std::endl;
        return false;
    }

    std::vector<int64_t> shape_information = d->output_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType type_information = d->output_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementType();
    assert(0 != shape_information.size());

    size_t total_element = d->output_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementCount();

    if (total_element == array_size) {
        DEBUG_EXECUTE(std::cout << "Total number of data checked same " << std::endl);
    } else {
        std::cout << "Total number of data checked incompatible, [Inner: " << total_element << "] vs [Outer: " 
        << array_size <<"]" << std::endl;
        return false;
    }




    if (deduce_type<T>() == type_information) {
        switch (layout) {
            case Predictor::RowMajor: {
                T* buff_in = d->output_tensors[tensor_index].GetTensorMutableData<T>();
                std::copy_n(buff_in, d->output_tensors[tensor_index].GetTensorTypeAndShapeInfo().GetElementCount(), p_data);
                return true;
            } break;
            case Predictor::ColumnMajor: {
                get_tensor_data_col_eigen_sametype<T>(&(d->output_tensors[tensor_index]), p_data);
                return true;
            } break;

            default: {
                // std::cout << "Different datatype between tensor definition and data source, auto casting enabled." << std::endl;
            } break;
        }
    }
    DEBUG_EXECUTE(std::cout << "Different datatype between tensor definition and data source, auto casting enabled." << std::endl);

    switch (method) {
        case Predictor::Eigen: {
            switch (layout) {
                case Predictor::RowMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            get_tensor_data_row_eigen<float, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            get_tensor_data_row_eigen<double, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            get_tensor_data_row_eigen<int32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            get_tensor_data_row_eigen<int16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            get_tensor_data_row_eigen<int8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            get_tensor_data_row_eigen<uint32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            get_tensor_data_row_eigen<uint16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            get_tensor_data_row_eigen<uint8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            get_tensor_data_col_eigen<float, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            get_tensor_data_col_eigen<double, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            get_tensor_data_col_eigen<int32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            get_tensor_data_col_eigen<int16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            get_tensor_data_col_eigen<int8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            get_tensor_data_col_eigen<uint32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            get_tensor_data_col_eigen<uint16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            get_tensor_data_col_eigen<uint8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                default: {
                    std::cout << "fatal: Unsupported layout, get data didn't succeed." << std::endl;
                    return false;
                } break;
            }
        } break;

        case Predictor::Simple: {
            switch (layout) {
                case Predictor::RowMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            get_tensor_data_row_simple<float, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            get_tensor_data_row_simple<double, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            get_tensor_data_row_simple<int32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            get_tensor_data_row_simple<int16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            get_tensor_data_row_simple<int8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            get_tensor_data_row_simple<uint32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            get_tensor_data_row_simple<uint16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            get_tensor_data_row_simple<uint8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                            get_tensor_data_col_simple<float, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                            get_tensor_data_col_simple<double, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                            get_tensor_data_col_simple<int32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                            get_tensor_data_col_simple<int16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                            get_tensor_data_col_simple<int8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                            get_tensor_data_col_simple<uint32_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                            get_tensor_data_col_simple<uint16_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;
                        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                            get_tensor_data_col_simple<uint8_t, T>(&(d->output_tensors[tensor_index]), p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                default: {
                    std::cout << "fatal: Unsupported layout, get data didn't succeed." << std::endl;
                    return false;
                } break;
            }
        } break;

        default: {
            std::cout << "fatal: Unsupported method, get data didn't succeed." << std::endl;
            return false;
        } break;
    }
    return true;
}

// TODO: instantiate template function
template bool Predictor::get_node_data<float>(std::string name, float* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<double>(std::string name, double* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Predictor::DataLayout layout, Predictor::CopyMethod method);

//-------------------------------------------------------------------------------------------------
//Wrapper for std::vector

template <typename T>
bool Predictor::get_node_data(std::string name, std::vector<T>& data) {
    return this->get_node_data(name, data.data(), data.size(), Predictor::RowMajor, Predictor::Eigen);
}

template bool Predictor::get_node_data<float>(std::string name, std::vector<float>& data);
template bool Predictor::get_node_data<double>(std::string name, std::vector<double>& data);
template bool Predictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data);
template bool Predictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data);
template bool Predictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data);
template bool Predictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data);
template bool Predictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data);
template bool Predictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data);

template <typename T>
bool Predictor::get_node_data(std::string name, std::vector<T>& data, Predictor::DataLayout layout) {
    return this->get_node_data(name, data.data(), data.size(), layout, Predictor::Eigen);
}

template bool Predictor::get_node_data<float>(std::string name, std::vector<float>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<double>(std::string name, std::vector<double>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Predictor::DataLayout layout);
template bool Predictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Predictor::DataLayout layout);
// template bool Predictor::get_node_data<uint64_t>(std::string name, std::vector<uint64_t>& data, Predictor::DataLayout layout);
// template bool Predictor::get_node_data<int64_t>(std::string name, std::vector<int64_t>& data, Predictor::DataLayout layout);

template <typename T>
bool Predictor::get_node_data(std::string name, std::vector<T>& data, Predictor::DataLayout layout, Predictor::CopyMethod method) {
    return this->get_node_data(name, data.data(), data.size(), layout, method);
}
// TODO: instantiate template function
template bool Predictor::get_node_data<float>(std::string name, std::vector<float>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<double>(std::string name, std::vector<double>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);
template bool Predictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Predictor::DataLayout layout, Predictor::CopyMethod method);

// --------------------------------------------------------------------------------------------------------------------
void Predictor::run() {
    if (d->input_tensors.size() == 0) {
        std::cerr << "Input tensors object not created" << std::endl;
        std::cerr << "Failed to Run session, please check if the data count has been set." << std::endl;
        return;
    }
    if (d->output_tensors.size() == 0) {
        std::cerr << "Output tensors object not created" << std::endl;
        std::cerr << "Failed to Run session, please check if the data count has been set." << std::endl;
        return;
    }

    // d->model.session->Run(Ort::RunOptions{ nullptr }, // e.g. set a verbosity level only for this run
    //     d->input_names.data(),  d->input_tensors.data(),  d->input_tensors.size(), // input to set
    //     d->output_names.data(), d->output_tensors.data(), d->output_tensors.size()); 
    d->model.session->Run(*(d->model.runoptions), // e.g. set a verbosity level only for this run
        d->input_names.data(),  d->input_tensors.data(),  d->input_tensors.size(), // input to set
        d->output_names.data(), d->output_tensors.data(), d->output_tensors.size()); 

}

// NodeInfo PredictorImpl::get_info_from_model(std::string name, bool& is_node_legal) {
//     NodeInfo info;
//     std::cout << "Start detecting node info from graph with name: " << name << std::endl;

//     const char* name_char = name.data();

//     TF_Operation* oper = TF_GraphOperationByName(this->model.graph, name_char);

//     if (oper == nullptr) {
//         std::cerr << "Node with name: " << name << " does not exist." << std::endl;
//         return info;
//     }

//     std::cout << "Node: {" << TF_OperationName(oper) << "} Info:" << std::endl;
//     std::cout << "--------------------------------------" << std::endl;

//     if (TF_OperationNumOutputs(oper) <= 0) {
//         is_node_legal = false;
//         std::cerr << "Node with name: " << name << " has no output." << std::endl;
//         return info;
//     }

//     info.op = {oper, 0};

//     // assert((TF_OperationOutputType(output) > 0) && (TF_OperationOutputType(output) < 25));

//     if (this->DT_TO_STRING.find(TF_OperationOutputType(info.op)) == this->DT_TO_STRING.end()) {
//         std::cout << "Datatype:             " << " UNKNOWN " << std::endl;
//     } else {
//         std::cout << "Datatype:             " << this->DT_TO_STRING[TF_OperationOutputType(info.op)] << std::endl;
//     }

//     info.type = TF_OperationOutputType(info.op);

//     int num_dim = TF_GraphGetTensorNumDims(this->model.graph, info.op, this->model.status);

//     if (TF_GetCode(this->model.status) != TF_OK) {
//         std::cerr << TF_Message(this->model.status) << std::endl;
//         is_node_legal = false;
//         return info;
//     }

//     std::vector<int64_t> shape_arr;

//     if (num_dim < 0) {
//         std::cerr << "Unable to detect num_dim from node: " << TF_OperationName(oper) << std::endl;
//         is_node_legal = false;
//         return info;
//     } else {
//         shape_arr.resize(num_dim);
//         TF_GraphGetTensorShape(this->model.graph,
//                                info.op,
//                                shape_arr.data(), num_dim,
//                                this->model.status);
//         if (TF_GetCode(this->model.status) != TF_OK) {
//             std::cerr << TF_Message(this->model.status) << std::endl;
//             is_node_legal = false;
//             return info;
//         }
//         info.shape = shape_arr;
//     }



//     std::cout << "Tensor_shape:         ";
//     print_shape(shape_arr);
//     std::cout << "--------------------------------------\n" << std::endl;


//     is_node_legal = true;
//     return info;
// }

// void NoOpDeallocator(void* data, size_t, void*) {}

// std::vector<tensorflow::int64> get_shape_from_tensor(tensorflow::Tensor& tensor) {
//     std::vector<tensorflow::int64> shape;
//     int64_t num_dimensions = tensor.shape().dims();
//     for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
//         shape.push_back(tensor.shape().dim_size(ii_dim));
//     }
//     return shape;
// }
