#include "predictor.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <limits>

#include "tensorflow/c/c_api.h"

#include <unsupported/Eigen/CXX11/Tensor>

#ifdef _DEBUG
#define DEBUG_EXECUTE(x) (x)
#else
#define DEBUG_EXECUTE(x) do {} while (0)
#endif



template <typename T>
TF_DataType deduce_type() {
    if (std::is_same<T, float>::value)
        return TF_FLOAT;
    if (std::is_same<T, double>::value)
        return TF_DOUBLE;
    if (std::is_same<T, int32_t>::value)
        return TF_INT32;
    if (std::is_same<T, uint8_t>::value)
        return TF_UINT8;
    if (std::is_same<T, int16_t>::value)
        return TF_INT16;
    if (std::is_same<T, int8_t>::value)
        return TF_INT8;
    if (std::is_same<T, int64_t>::value)
        return TF_INT64;
    if (std::is_same<T, uint16_t>::value)
        return TF_UINT16;
    if (std::is_same<T, uint32_t>::value)
        return TF_UINT32;
    if (std::is_same<T, uint64_t>::value)
        return TF_UINT64;

    std::cerr << "Not supported data type." << std::endl;
    return static_cast<TF_DataType>(-10);
}


// instantiate template functions
template TF_DataType deduce_type<float>();
template TF_DataType deduce_type<double>();
template TF_DataType deduce_type<int8_t>();
template TF_DataType deduce_type<int16_t>();
template TF_DataType deduce_type<int32_t>();
template TF_DataType deduce_type<int64_t>();
template TF_DataType deduce_type<uint8_t>();
template TF_DataType deduce_type<uint16_t>();
template TF_DataType deduce_type<uint32_t>();
template TF_DataType deduce_type<uint64_t>();



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
    std::vector<int64_t>           shape;
    TF_DataType                    type;
    TF_Tensor*                     tensor;
    TF_Output                      op;
    NodeInfo()
        : tensor(nullptr), op({nullptr, 0}),type(static_cast<TF_DataType>(-1)){};
};


struct TFModel {
    TF_Status*                         status;
    TF_Session*                        session;
    TF_Graph*                          graph;
    TFModel()
        : status(nullptr), session(nullptr), graph(nullptr) {};
};


// read all content of file into tensor buffer
TF_Buffer* read_file(const std::string& filename) {
    std::ifstream pbfile(filename, std::ios::binary);

    if (!pbfile.is_open()) {
        std::cerr << "Unable to open pbfile: " << filename << std::endl;
        return nullptr;
    }

    // get file size
    pbfile.seekg(0, std::ios::end);
    auto size = pbfile.tellg();
    pbfile.seekg(0, std::ios::beg);

    // read file content
    auto data = new char[size];
    pbfile.read(data, size);
    pbfile.close();

    // create tensorflow buffer from read data
    TF_Buffer* buffer = TF_NewBufferFromString(data, size);
    delete[] data;
    return buffer;
}



class PredictorImpl {
public:
    int64_t                           data_count;
    std::map<std::string, NodeInfo>   input_nodes;
    std::map<std::string, NodeInfo>   output_nodes;

    TFModel model;

    NodeInfo get_info_from_model(std::string name, bool& is_node_legal);

    std::map<int, std::string> DT_TO_STRING = {
        { 0,  "DT_INVALID"    },
        { 1,  "DT_FLOAT"      },
        { 2,  "DT_DOUBLE"     },
        { 3,  "DT_INT32"      },
        { 4,  "DT_UINT8"      },
        { 5,  "DT_INT16"      },
        { 6,  "DT_INT8"       },
        { 7,  "DT_STRING"     },
        { 8,  "DT_COMPLEX64"  },
        { 9,  "DT_INT64"      },
        { 10, "DT_BOOL"       },
        { 11, "DT_QINT8"      },
        { 12, "DT_QUINT8"     },
        { 13, "DT_QINT32"     },
        { 14, "DT_BFLOAT16"   },
        { 15, "DT_QINT16"     },
        { 16, "DT_QUINT16"    },
        { 17, "DT_UINT16"     },
        { 18, "DT_COMPLEX128" },
        { 19, "DT_HALF"       },
        { 20, "DT_RESOURCE"   },
        { 21, "DT_VARIANT"    },
        { 22, "DT_UINT32"     },
        { 23, "DT_UINT64"     }
    };
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
    assert((intra_op_parallelism_threads >= 0) && (inter_op_parallelism_threads >= 0));

    int uint8lim =  static_cast<int>(std::numeric_limits<uint8_t>::max());
    assert((intra_op_parallelism_threads <= uint8lim) && (inter_op_parallelism_threads <= uint8lim));
    uint8_t intra_threads = static_cast<uint8_t>(intra_op_parallelism_threads);
    uint8_t inter_threads = static_cast<uint8_t>(inter_op_parallelism_threads);



    d = new PredictorImpl();
    d->data_count = -1;

    std::cout << "Initializing model from frozen graph (*.pb)..." << std::endl;
    d->model.status = TF_NewStatus();
    d->model.graph = TF_NewGraph();

    std::cout << "Reading Tensorflow GraphDef..." << std::endl;
    TF_Buffer* graph_def = read_file(pbfile);

    if (graph_def == nullptr) {
        std::cerr << "Failed to read pb file." << std::endl;
        return;
    }

    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(d->model.graph, graph_def, import_opts, d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }

    TF_DeleteImportGraphDefOptions(import_opts);
    TF_DeleteBuffer(graph_def);

    std::cout << "Creating Tensorflow Session..." << std::endl;
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();

    std::cout << "Setting Paralism option ..." << std::endl;
    std::cout << "intra_op_parallelism_threads = " << static_cast<int>(intra_threads) << std::endl;
    std::cout << "inter_op_parallelism_threads = " << static_cast<int>(inter_threads) << std::endl;
    uint8_t config_buf[] = {0x10, intra_threads, 0x28, inter_threads};

    TF_SetConfig(sess_opts, config_buf, sizeof(config_buf), d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }

    d->model.session = TF_NewSession(d->model.graph, sess_opts, d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }
    TF_DeleteSessionOptions(sess_opts);
}


Predictor::Predictor(std::string folder, std::string tag, int intra_op_parallelism_threads, int inter_op_parallelism_threads)
    : d(nullptr) {

    assert((intra_op_parallelism_threads >= 0) && (inter_op_parallelism_threads >= 0));

    int uint8lim =  static_cast<int>(std::numeric_limits<uint8_t>::max());
    assert((intra_op_parallelism_threads <= uint8lim) && (inter_op_parallelism_threads <= uint8lim));
    uint8_t intra_threads = static_cast<uint8_t>(intra_op_parallelism_threads);
    uint8_t inter_threads = static_cast<uint8_t>(inter_op_parallelism_threads);

    
    d = new PredictorImpl();
    d->data_count = -1;

    std::cout << "Initializing model from saved model..." << std::endl;
    d->model.status = TF_NewStatus();
    d->model.graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();

    std::cout << "Setting Paralism option ..." << std::endl;
    std::cout << "intra_op_parallelism_threads = " << static_cast<int>(intra_threads) << std::endl;
    std::cout << "inter_op_parallelism_threads = " << static_cast<int>(inter_threads) << std::endl;
    uint8_t config_buf[] = {0x10, intra_threads, 0x28, inter_threads};

    TF_SetConfig(sess_opts, config_buf, sizeof(config_buf), d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }

    std::cout << "Loading saved model..." << std::endl;
    const char* tag_char = tag.data();
    d->model.session = TF_LoadSessionFromSavedModel(sess_opts, nullptr, folder.data(), &tag_char, 1, d->model.graph, nullptr, d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }
    TF_DeleteSessionOptions(sess_opts);
}

//----------------------------------------------Destructor--------------------------------------

Predictor::~Predictor() {
    for (auto i = d->input_nodes.begin(); i != d->input_nodes.end(); ++i) {
        if (nullptr != i->second.tensor) {
            TF_DeleteTensor(i->second.tensor);
        }
    }
    for (auto i = d->output_nodes.begin(); i != d->output_nodes.end(); ++i) {
        if (nullptr != i->second.tensor) {
            TF_DeleteTensor(i->second.tensor);
        }
    }

    if (nullptr != d->model.session) {
        TF_DeleteSession(d->model.session, d->model.status);
        if (TF_GetCode(d->model.status) != TF_OK) {
            std::cerr << TF_Message(d->model.status) << std::endl;
            return;
        }
    }

    if (nullptr != d->model.graph) {
        TF_DeleteGraph(d->model.graph);
    }
    if (nullptr != d->model.status) {
        TF_DeleteStatus(d->model.status);
    }

    delete d;
}

void Predictor::print_operations(std::string node_name) {
    if (d->model.graph == nullptr) {
        std::cerr << "Model is not properly initialized, please check your initialization." << std::endl;
        return;
    }

    const char* name_char = node_name.data();

    TF_Operation* oper = TF_GraphOperationByName(d->model.graph, name_char);

    if (oper == nullptr) {
        std::cerr << "Node with name: " << node_name << "does not exist." << std::endl;
        return;
    }

    std::cout << std::endl;
    std::cout << "Node: {" << TF_OperationName(oper) << "} Info" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Operation_type:       " << TF_OperationOpType(oper) << std::endl;

    if (TF_OperationNumOutputs(oper) <= 0) {
        return;
    }

    TF_Output output = {oper, 0};
    int num_dim = TF_GraphGetTensorNumDims(d->model.graph, output, d->model.status);

    if (TF_GetCode(d->model.status) != TF_OK) {
        std::cerr << TF_Message(d->model.status) << std::endl;
        return;
    }

    std::vector<int64_t> shape_arr;

    if (num_dim < 0) {
        std::cerr << "Unable to detect num_dim from node: " << TF_OperationName(oper) << std::endl;
        return;
    } else {
        shape_arr.resize(num_dim);
        TF_GraphGetTensorShape(d->model.graph,
                               output,
                               shape_arr.data(), num_dim,
                               d->model.status);
        if (TF_GetCode(d->model.status) != TF_OK) {
            std::cerr << TF_Message(d->model.status) << std::endl;
        }
    }

    if (d->DT_TO_STRING.find(TF_OperationOutputType(output)) == d->DT_TO_STRING.end()) {
        std::cout << "Datatype:             "
                  << " UNKNOWN " << std::endl;
    } else {
        std::cout << "Datatype:             " << d->DT_TO_STRING[TF_OperationOutputType(output)] << std::endl;
    }

    std::cout << "Tensor_shape:         ";
    print_shape(shape_arr);
    std::cout << "--------------------------------------" << std::endl;

}



void Predictor::print_operations() {
    if (d->model.graph == nullptr) {
        std::cout << "Model is not properly initialized, please check your initialization." << std::endl;
        return;
    }

    size_t pos = 0;
    TF_Operation* oper;

    while ((oper = TF_GraphNextOperation(d->model.graph, &pos)) != nullptr) {
        std::cout << std::endl;
        // std::cout << "--------------------------------------" << std::endl;
        std::cout << "Node: {" << TF_OperationName(oper) << "}:" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Operation_type:       " << TF_OperationOpType(oper) << std::endl;
        // std::cout << "Data_type:            " << d->DT_TO_STRING[TF_OperationOutputType(output)] << std::endl;

        if (TF_OperationNumOutputs(oper) <= 0) {
            continue;
        }

        TF_Output output = {oper, 0};
        int num_dim = TF_GraphGetTensorNumDims(d->model.graph, output, d->model.status);

        if (TF_GetCode(d->model.status) != TF_OK) {
            std::cerr << TF_Message(d->model.status) << std::endl;
            continue;
        }

        std::vector<int64_t> shape_arr;

        if (num_dim < 0) {
            std::cerr << "Unable to detect num_dim from node: " << TF_OperationName(oper) << std::endl;
            continue;
        } else {
            shape_arr.resize(num_dim);
            TF_GraphGetTensorShape(d->model.graph,
                                   output,
                                   shape_arr.data(), num_dim,
                                   d->model.status);
            if (TF_GetCode(d->model.status) != TF_OK) {
                std::cerr << TF_Message(d->model.status) << std::endl;
            }
        }

        if (d->DT_TO_STRING.find(TF_OperationOutputType(output)) == d->DT_TO_STRING.end()) {
            std::cout << "Datatype:             " << " UNKNOWN " << std::endl;
        } else {
            std::cout << "Datatype:             " << d->DT_TO_STRING[TF_OperationOutputType(output)] << std::endl;
        }

        std::cout << "Tensor_shape:         ";
        print_shape(shape_arr);
        std::cout << "--------------------------------------" << std::endl;
    }

}

void Predictor::regist_node(std::string name, Predictor::NodeType type) {
    switch (type) {
        case Predictor::INPUT_NODE: {
            if ((d->input_nodes.end() == d->input_nodes.find(name) ) &&
                (d->output_nodes.end() == d->output_nodes.find(name))) {
                std::cout << "Registering node: [ " << name << " ] as input:" << std::endl;
                bool is_node_legal = false;
                NodeInfo info = d->get_info_from_model(name, is_node_legal);
                if (is_node_legal) {
                    d->input_nodes.insert(std::make_pair(name, info));
                } else {
                    std::cout << "The node with name: [ " << name << " ] is not legal to be registered" << std::endl;
                    return;
                }
            } else {
                std::cout << "Node: [ " << name << "] has already been registered." << std::endl;
                return;
            }
        } break;
        case Predictor::OUTPUT_NODE: {
            if ((d->input_nodes.end() == d->input_nodes.find(name)) &&
                (d->output_nodes.end() == d->output_nodes.find(name))) {
                std::cout << "Registering node: [ " << name << " ] as output:" << std::endl;
                bool is_node_legal = false;
                NodeInfo info = d->get_info_from_model(name, is_node_legal);
                if (is_node_legal) {
                    d->output_nodes.insert(std::make_pair(name, info));
                } else {
                    std::cout << "The node with name: [ " << name << " ] is not legal to be registered" << std::endl;
                    return;
                }
            } else {
                std::cout << "Node: [ " << name << " ] has already been registered." << std::endl;
                return;
            }
        } break;
        default:
            std::cerr << "Please specify node type: {Predictor::INPUT_NODE} or {Predictor::OUTPUT_NODE}" << std::endl;
            return;
    }
}

void Predictor::set_data_count(int64_t cnt) {
    assert(cnt > 0);

    if (-1 != d->data_count) {
        DEBUG_EXECUTE(std::cerr << "warning: overwriting data count" << std::endl);
    }
    d->data_count = cnt;
}

int64_t Predictor::get_data_count() {
    return d->data_count;
}

//-------------------------------------------------------------------------------------------------
template <typename T_data, typename T_tensor>
static void set_tensor_data_row_simple(T_data* src, TF_Tensor* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_row_simple" << std::endl);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(dst));
    int nelems = TF_TensorElementCount(dst);
    for (int i = 0; i < nelems; ++i) {
        buff_tensor[i] = src[i];
    }
}

/// @brief partial transpose dst[i,j,k, :] = src[:, k,j,i].T
/// @param rank         number of dimensions
/// @param slice_rank   number of fixed dimensions (known index)
template <typename T, typename U>
static void transpose_slice(int rank, int slice_rank, const int dst_shape[],
                            T* dst, const int dst_step[], int dst_offset,
                            const U* src, const int src_step[], int src_offset) {
    int limit = dst_shape[slice_rank];
    int dstep = dst_step[slice_rank];
    int sstep = src_step[rank - slice_rank - 1];

    if (slice_rank == rank - 1) {
        for (int i = 0; i < limit; ++i) {
            dst[dst_offset] = src[src_offset];
            dst_offset += dstep;
            src_offset += sstep;
        }
        return;
    }

    for (int i = 0; i < limit; ++i) {
        transpose_slice(rank, slice_rank + 1, dst_shape,
            dst, dst_step, dst_offset,
            src, src_step, src_offset);
        dst_offset += dstep;
        src_offset += sstep;
    }
}

template <typename T, typename U>
static void simple_transpose(const U* src, int rank, const int dims[], T* dst) {

    // get tensor shape
    const int* src_shape = dims;
    int* dst_shape = new int[rank];
    for (int i = 0; i < rank; ++i) {
        dst_shape[i] = src_shape[rank - i - 1];
    }

    // get step of each dimension
    int *dst_step = new int[rank];
    int *src_step = new int[rank];
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


/*
ndarray dim [d4, d3, d2, d1] = [5,6,7,8]
slice [a, :, c, d]
offset, step

step(2) = [d3*d2*d1, d2*d1, d1, 1]
[a,b,c,d] -- [a,b,c+1, d]

flatten (row-major)
[0,0,0,0]~[0,0,0,7]
[0,0,1,0]~[0,0,1,7]

dst = src.T (N)

dst[0, :] = src[:, 0].T (N-1)
dst[1, :] = src[:, 1].T
dst[2, :] = src[:, 2].T
dst[3, :] = src[:, 3].T

*/

// multidimension transpose set
// dst[i,j,k,...] = src[...,k,j,i]
template <typename T_data, typename T_tensor>
static void set_tensor_data_col_simple(T_data* src, TF_Tensor* dst) {
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(dst));

    int rank = TF_NumDims(dst);
    if (1 == rank) {
        set_tensor_data_row_simple<T_data, T_tensor>(src, dst);
        return;
    }

    int* src_shape = new int[rank];
    for (int i = 0; i < rank; ++i) {
        src_shape[i] = TF_Dim(dst, rank - i - 1);
    }

    simple_transpose<T_tensor, T_data>(src, rank, src_shape, buff_tensor);
    delete[] src_shape;
}

template <typename T_data, typename T_tensor>
static void set_tensor_data_row_eigen(T_data* src, TF_Tensor* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_row_eigen" << std::endl);
    Eigen::array<Eigen::Index, 1> shape_arr({TF_TensorElementCount(dst)});
    Eigen::TensorMap<Eigen::Tensor<T_data, 1, Eigen::RowMajor>> data_wrapper(src, shape_arr);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(dst));
    Eigen::TensorMap<Eigen::Tensor<T_tensor, 1, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_arr);
    tensor_wrapper = data_wrapper.template cast<T_tensor>();
}

template <typename T_data, typename T_tensor>
static void set_tensor_data_col_eigen(T_data* src, TF_Tensor* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_col_eigen" << std::endl);
    int rank = TF_NumDims(dst);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(dst));
    switch (rank) {
        case 1: {
            set_tensor_data_row_eigen<T_data, T_tensor>(src, dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1)});
            Eigen::TensorMap<Eigen::Tensor<T_tensor, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);
            Eigen::array<int, 2> shuffling({1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({TF_Dim(dst, 4),
                                                      TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3),
                                                        TF_Dim(dst, 4)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template cast<T_tensor>().template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({TF_Dim(dst, 5),
                                                      TF_Dim(dst, 4),
                                                      TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3),
                                                        TF_Dim(dst, 4),
                                                        TF_Dim(dst, 5)});

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
}

template <typename T_data>
static void set_tensor_data_col_eigen_sametype(T_data* src, TF_Tensor* dst) {
    DEBUG_EXECUTE(std::cout << "set_tensor_data_col_eigen_sametype" << std::endl);
    int rank = TF_NumDims(dst);
    T_data* buff_tensor = static_cast<T_data*>(TF_TensorData(dst));
    switch (rank) {
        case 1: {
            std::copy_n(src, TF_TensorElementCount(dst), buff_tensor);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 2> shuffling({1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(src, shape_data);


            Eigen::array<Eigen::Index, 3> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({TF_Dim(dst, 4),
                                                      TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3),
                                                        TF_Dim(dst, 4)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            tensor_wrapper = data_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({TF_Dim(dst, 5),
                                                      TF_Dim(dst, 4),
                                                      TF_Dim(dst, 3),
                                                      TF_Dim(dst, 2),
                                                      TF_Dim(dst, 1),
                                                      TF_Dim(dst, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(src, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({TF_Dim(dst, 0),
                                                        TF_Dim(dst, 1),
                                                        TF_Dim(dst, 2),
                                                        TF_Dim(dst, 3),
                                                        TF_Dim(dst, 4),
                                                        TF_Dim(dst, 5)});

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
    if (d->input_nodes.end() != d->input_nodes.find(name)) {
        auto it = d->input_nodes.find(name);
        std::vector<int64_t> shape_information = it->second.shape;
        TF_DataType type_information = it->second.type;
        assert(0 != shape_information.size());

        for (int i = 0; i < shape_information.size(); i++) {
            if (shape_information[i] <= 0) {
                // std::cout << "Setting unknown dim rank: " << i << " to given data length: " << d->data_count << std::endl;
                shape_information[i] = d->data_count;
                assert(shape_information[i] > 0);
            }
        }

        int total_element = get_numelements_from_shape(shape_information);

        if (nullptr == it->second.tensor) {
            DEBUG_EXECUTE(std::cout << "Tensor is not initiated, creating tensor obj..." << std::endl);

            if (total_element <=0) {
                std::cout << "Detected tensor with unspecified dimension, please set data count, Failed set data" << std::endl;
                return false;
            }
            it->second.tensor = TF_AllocateTensor(type_information, shape_information.data(), shape_information.size(), total_element * TF_DataTypeSize(type_information));

        } else if (get_numelements_from_shape(shape_information) != TF_TensorElementCount(it->second.tensor) ) {
            DEBUG_EXECUTE(std::cout << "Checked size changed, creating new tensor obj ..." << std::endl);
            TF_DeleteTensor(it->second.tensor);
            it->second.tensor = TF_AllocateTensor(type_information, shape_information.data(), shape_information.size(), total_element * TF_DataTypeSize(type_information));
        }

        if (TF_TensorElementCount(it->second.tensor) == array_size) {
            DEBUG_EXECUTE(std::cout << "Total number of data checked same " << std::endl);
        } else {
            std::cout << "Total number of data checked incompatible: " << TF_TensorElementCount(it->second.tensor) << " vs " << array_size << std::endl;
            return false;
        }

        if (deduce_type<T>() == type_information) {
            switch (layout) {
                case Predictor::RowMajor: {
                    T* buff_in = static_cast<T*>(TF_TensorData(it->second.tensor));
                    std::copy_n(p_data, TF_TensorElementCount(it->second.tensor), buff_in);
                    return true;
                } break;
                case Predictor::ColumnMajor: {
                    set_tensor_data_col_eigen_sametype<T>(p_data, it->second.tensor);
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
                            case TF_FLOAT:
                                set_tensor_data_row_eigen<T, float>(p_data, it->second.tensor);
                                break;
                            case TF_DOUBLE:
                                set_tensor_data_row_eigen<T, double>(p_data, it->second.tensor);
                                break;
                            case TF_INT32:
                                set_tensor_data_row_eigen<T, int32_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT16:
                                set_tensor_data_row_eigen<T, int16_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT8:
                                set_tensor_data_row_eigen<T, int8_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT32:
                                set_tensor_data_row_eigen<T, uint32_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT16:
                                set_tensor_data_row_eigen<T, uint16_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT8:
                                set_tensor_data_row_eigen<T, uint8_t>(p_data, it->second.tensor);
                                break;

                            default: {
                                std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                                return false;
                            } break;
                        }
                    } break;

                    case Predictor::ColumnMajor: {
                        switch (type_information) {
                            case TF_FLOAT:
                                set_tensor_data_col_eigen<T, float>(p_data, it->second.tensor);
                                break;
                            case TF_DOUBLE:
                                set_tensor_data_col_eigen<T, double>(p_data, it->second.tensor);
                                break;
                            case TF_INT32:
                                set_tensor_data_col_eigen<T, int32_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT16:
                                set_tensor_data_col_eigen<T, int16_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT8:
                                set_tensor_data_col_eigen<T, int8_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT32:
                                set_tensor_data_col_eigen<T, uint32_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT16:
                                set_tensor_data_col_eigen<T, uint16_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT8:
                                set_tensor_data_col_eigen<T, uint8_t>(p_data, it->second.tensor);
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
                            case TF_FLOAT:
                                set_tensor_data_row_simple<T, float>(p_data, it->second.tensor);
                                break;
                            case TF_DOUBLE:
                                set_tensor_data_row_simple<T, double>(p_data, it->second.tensor);
                                break;
                            case TF_INT32:
                                set_tensor_data_row_simple<T, int32_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT16:
                                set_tensor_data_row_simple<T, int16_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT8:
                                set_tensor_data_row_simple<T, int8_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT32:
                                set_tensor_data_row_simple<T, uint32_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT16:
                                set_tensor_data_row_simple<T, uint16_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT8:
                                set_tensor_data_row_simple<T, uint8_t>(p_data, it->second.tensor);
                                break;

                            default: {
                                std::cout << "fatal: Unsupported data type, set data didn't succeed." << std::endl;
                                return false;
                            } break;
                        }
                    } break;

                    case Predictor::ColumnMajor: {
                        switch (type_information) {
                            case TF_FLOAT:
                                set_tensor_data_col_simple<T, float>(p_data, it->second.tensor);
                                break;
                            case TF_DOUBLE:
                                set_tensor_data_col_simple<T, double>(p_data, it->second.tensor);
                                break;
                            case TF_INT32:
                                set_tensor_data_col_simple<T, int32_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT16:
                                set_tensor_data_col_simple<T, int16_t>(p_data, it->second.tensor);
                                break;
                            case TF_INT8:
                                set_tensor_data_col_simple<T, int8_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT32:
                                set_tensor_data_col_simple<T, uint32_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT16:
                                set_tensor_data_col_simple<T, uint16_t>(p_data, it->second.tensor);
                                break;
                            case TF_UINT8:
                                set_tensor_data_col_simple<T, uint8_t>(p_data, it->second.tensor);
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

    std::cerr << "fatal: input node " << name << " not registered!" << std::endl;
    return false;
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
static void get_tensor_data_row_simple(TF_Tensor* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_row_simple" << std::endl);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(src));
    for (int i = 0; i < TF_TensorElementCount(src); ++i) {
        dst[i] = buff_tensor[i];
    }
}

template <typename T_tensor, typename T_data>
static void get_tensor_data_col_simple(TF_Tensor* src, T_data* dst) {
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(src));

    int rank = TF_NumDims(src);
    if (1 == rank) {
        get_tensor_data_row_simple<T_tensor, T_data>(src, dst);
        return;
    }

    // get tensor shape
    // int* dst_shape = new int[rank];
    int* src_shape = new int[rank];
    for (int i = 0; i < rank; ++i) {
        src_shape[i] = TF_Dim(src, i);
    }

    simple_transpose<T_data, T_tensor>(buff_tensor, rank, src_shape, dst);
    delete[] src_shape;

}


template <typename T_tensor, typename T_data>
static void get_tensor_data_row_eigen(TF_Tensor* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_row_eigen" << std::endl);
    Eigen::array<Eigen::Index, 1> shape_arr({TF_TensorElementCount(src)});
    Eigen::TensorMap<Eigen::Tensor<T_data, 1, Eigen::RowMajor>> data_wrapper(dst, shape_arr);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(src));
    Eigen::TensorMap<Eigen::Tensor<T_tensor, 1, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_arr);
    data_wrapper = tensor_wrapper.template cast<T_data>();
}

template <typename T_tensor, typename T_data>
static void get_tensor_data_col_eigen(TF_Tensor* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_col_eigen" << std::endl);
    int rank = TF_NumDims(src);
    T_tensor* buff_tensor = static_cast<T_tensor*>(TF_TensorData(src));
    switch (rank) {
        case 1: {
            get_tensor_data_row_eigen<T_tensor, T_data>(src, dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1)});
            Eigen::TensorMap<Eigen::Tensor<T_tensor, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);
            Eigen::array<int, 2> shuffling({1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({TF_Dim(src, 4),
                                                      TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3),
                                                        TF_Dim(src, 4)});

            Eigen::TensorMap<Eigen::Tensor<T_tensor, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template cast<T_data>().template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({TF_Dim(src, 5),
                                                      TF_Dim(src, 4),
                                                      TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3),
                                                        TF_Dim(src, 4),
                                                        TF_Dim(src, 5)});

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
}

template <typename T_data>
static void get_tensor_data_col_eigen_sametype(TF_Tensor* src, T_data* dst) {
    DEBUG_EXECUTE(std::cout << "get_tensor_data_col_eigen_sametype" << std::endl);
    int rank = TF_NumDims(src);
    T_data* buff_tensor = static_cast<T_data*>(TF_TensorData(src));

    switch (rank) {
        case 1: {
            std::copy_n(buff_tensor, TF_TensorElementCount(src), dst);
            return;
        } break;

        case 2: {
            Eigen::array<Eigen::Index, 2> shape_data({TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 2> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 2, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 2> shuffling({1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 3: {
            Eigen::array<Eigen::Index, 3> shape_data({TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 3> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 3, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 3> shuffling({2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 4: {
            Eigen::array<Eigen::Index, 4> shape_data({TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 4> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 4, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 4> shuffling({3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 5: {
            Eigen::array<Eigen::Index, 5> shape_data({TF_Dim(src, 4),
                                                      TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 5> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3),
                                                        TF_Dim(src, 4)});

            Eigen::TensorMap<Eigen::Tensor<T_data, 5, Eigen::RowMajor>> tensor_wrapper(buff_tensor, shape_tensor);

            Eigen::array<int, 5> shuffling({4, 3, 2, 1, 0});
            data_wrapper = tensor_wrapper.template shuffle(shuffling);
            return;
        } break;

        case 6: {
            Eigen::array<Eigen::Index, 6> shape_data({TF_Dim(src, 5),
                                                      TF_Dim(src, 4),
                                                      TF_Dim(src, 3),
                                                      TF_Dim(src, 2),
                                                      TF_Dim(src, 1),
                                                      TF_Dim(src, 0)});
            Eigen::TensorMap<Eigen::Tensor<T_data, 6, Eigen::RowMajor>> data_wrapper(dst, shape_data);

            Eigen::array<Eigen::Index, 6> shape_tensor({TF_Dim(src, 0),
                                                        TF_Dim(src, 1),
                                                        TF_Dim(src, 2),
                                                        TF_Dim(src, 3),
                                                        TF_Dim(src, 4),
                                                        TF_Dim(src, 5)});

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
    auto it = d->output_nodes.find(name);

    if (d->output_nodes.end() != d->output_nodes.find(name)) {
        DEBUG_EXECUTE(std::cout << "Detected registered node: " << name << " as output to extract data." << std::endl);
        it = d->output_nodes.find(name);
    } else if (d->input_nodes.end() != d->input_nodes.find(name)) {
        DEBUG_EXECUTE(std::cout << "Detected registered node: " << name << " as input to extract data." << std::endl);
        it = d->input_nodes.find(name);
    } else {
        std::cerr << "fatal: node " << name << " not registered!" << std::endl;
        return false;
    }

    std::vector<int64_t> shape_information = it->second.shape;
    TF_DataType type_information = it->second.type;
    // NodeInfo info = d->output_nodes[name];
    if (0 == shape_information.size()) {
        std::cerr << "fatal: output node " << name << " shaped not detected!" << std::endl;
        return false;
    }

    for (int i = 0; i < shape_information.size(); i++) {
        if (shape_information[i] <= 0) {
            DEBUG_EXECUTE(std::cout << "Setting unknown dim rank: " << i << " to given data length: " << d->data_count << std::endl);
            shape_information[i] = d->data_count;
            assert(shape_information[i] > 0);
        }
    }

    if (nullptr == it->second.tensor) {
        std::cerr << "fatal: output node " << name << " has no data" << std::endl;
        return false;
    }

    // assert(TF_TensorElementCount(it->second.tensor) == array_size);
    if (TF_TensorElementCount(it->second.tensor) == array_size) {
        // std::cout << "Total number of data checked same " << std::endl;
    } else {
        std::cout << "Total number of data checked incompatible: " << TF_TensorElementCount(it->second.tensor) << " vs " << array_size << std::endl;
        return false;
    }

    if (deduce_type<T>() == type_information) {
        switch (layout) {
            case Predictor::RowMajor: {
                T* buff_in = static_cast<T*>(TF_TensorData(it->second.tensor));
                std::copy_n(buff_in, TF_TensorElementCount(it->second.tensor), p_data);
                return true;
            } break;
            case Predictor::ColumnMajor: {
                get_tensor_data_col_eigen_sametype<T>(it->second.tensor, p_data);
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
                        case TF_FLOAT:
                            get_tensor_data_row_eigen<float, T>(it->second.tensor, p_data);
                            break;
                        case TF_DOUBLE:
                            get_tensor_data_row_eigen<double, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT32:
                            get_tensor_data_row_eigen<int32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT16:
                            get_tensor_data_row_eigen<int16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT8:
                            get_tensor_data_row_eigen<int8_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT32:
                            get_tensor_data_row_eigen<uint32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT16:
                            get_tensor_data_row_eigen<uint16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT8:
                            get_tensor_data_row_eigen<uint8_t, T>(it->second.tensor, p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case TF_FLOAT:
                            get_tensor_data_col_eigen<float, T>(it->second.tensor, p_data);
                            break;
                        case TF_DOUBLE:
                            get_tensor_data_col_eigen<double, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT32:
                            get_tensor_data_col_eigen<int32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT16:
                            get_tensor_data_col_eigen<int16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT8:
                            get_tensor_data_col_eigen<int8_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT32:
                            get_tensor_data_col_eigen<uint32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT16:
                            get_tensor_data_col_eigen<uint16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT8:
                            get_tensor_data_col_eigen<uint8_t, T>(it->second.tensor, p_data);
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
                        case TF_FLOAT:
                            get_tensor_data_row_simple<float, T>(it->second.tensor, p_data);
                            break;
                        case TF_DOUBLE:
                            get_tensor_data_row_simple<double, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT32:
                            get_tensor_data_row_simple<int32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT16:
                            get_tensor_data_row_simple<int16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT8:
                            get_tensor_data_row_simple<int8_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT32:
                            get_tensor_data_row_simple<uint32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT16:
                            get_tensor_data_row_simple<uint16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT8:
                            get_tensor_data_row_simple<uint8_t, T>(it->second.tensor, p_data);
                            break;

                        default: {
                            std::cout << "fatal: Unsupported data type, get data didn't succeed." << std::endl;
                            return false;
                        } break;
                    }
                } break;

                case Predictor::ColumnMajor: {
                    switch (type_information) {
                        case TF_FLOAT:
                            get_tensor_data_col_simple<float, T>(it->second.tensor, p_data);
                            break;
                        case TF_DOUBLE:
                            get_tensor_data_col_simple<double, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT32:
                            get_tensor_data_col_simple<int32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT16:
                            get_tensor_data_col_simple<int16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_INT8:
                            get_tensor_data_col_simple<int8_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT32:
                            get_tensor_data_col_simple<uint32_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT16:
                            get_tensor_data_col_simple<uint16_t, T>(it->second.tensor, p_data);
                            break;
                        case TF_UINT8:
                            get_tensor_data_col_simple<uint8_t, T>(it->second.tensor, p_data);
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
    std::vector<TF_Output> input_ops;
    input_ops.reserve(d->input_nodes.size());

    std::vector<TF_Output> output_ops;
    output_ops.reserve(d->output_nodes.size());

    std::vector<TF_Tensor*> input_tensors;
    input_tensors.reserve(d->input_nodes.size());

    std::vector<TF_Tensor*> output_tensors(d->output_nodes.size());


    for (auto i = d->input_nodes.begin(); i != d->input_nodes.end(); ++i) {
        if (i->second.tensor == nullptr) {
            std::cout << "Input node: " << i->first << " tensor object not created" << std::endl;
            std::cout << "Failed to Run session, please check if all inputs node has been set data." << std::endl;
            return;
        }

        input_ops.push_back(i->second.op);
        input_tensors.push_back(i->second.tensor);

    }

    for (auto i = d->output_nodes.begin(); i != d->output_nodes.end(); ++i) {
        output_ops.push_back(i->second.op);
    }

    TF_SessionRun(d->model.session, nullptr, input_ops.data(), input_tensors.data(), input_ops.size(), output_ops.data(), output_tensors.data(), output_ops.size(), nullptr, 0, nullptr, d->model.status);



    for (int i = 0; i < output_ops.size(); ++i) {
        std::string key = TF_OperationName(output_ops[i].oper);

        if (d->output_nodes[key].tensor == nullptr) {
            DEBUG_EXECUTE(std::cout << "Creating new output tensors" << std::endl);
            d->output_nodes[key].tensor = output_tensors[i];
        } else {
            DEBUG_EXECUTE(std::cout << "Releasing former created tensors" << std::endl);
            TF_DeleteTensor(d->output_nodes[key].tensor);
            d->output_nodes[key].tensor = output_tensors[i];
        }
    }
}

NodeInfo PredictorImpl::get_info_from_model(std::string name, bool& is_node_legal) {
    NodeInfo info;
    std::cout << "Start detecting node info from graph with name: " << name << std::endl;

    const char* name_char = name.data();

    TF_Operation* oper = TF_GraphOperationByName(this->model.graph, name_char);

    if (oper == nullptr) {
        std::cerr << "Node with name: " << name << " does not exist." << std::endl;
        return info;
    }

    std::cout << "Node: {" << TF_OperationName(oper) << "} Info:" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    if (TF_OperationNumOutputs(oper) <= 0) {
        is_node_legal = false;
        std::cerr << "Node with name: " << name << " has no output." << std::endl;
        return info;
    }

    info.op = {oper, 0};

    // assert((TF_OperationOutputType(output) > 0) && (TF_OperationOutputType(output) < 25));

    if (this->DT_TO_STRING.find(TF_OperationOutputType(info.op)) == this->DT_TO_STRING.end()) {
        std::cout << "Datatype:             " << " UNKNOWN " << std::endl;
    } else {
        std::cout << "Datatype:             " << this->DT_TO_STRING[TF_OperationOutputType(info.op)] << std::endl;
    }

    info.type = TF_OperationOutputType(info.op);

    int num_dim = TF_GraphGetTensorNumDims(this->model.graph, info.op, this->model.status);

    if (TF_GetCode(this->model.status) != TF_OK) {
        std::cerr << TF_Message(this->model.status) << std::endl;
        is_node_legal = false;
        return info;
    }

    std::vector<int64_t> shape_arr;

    if (num_dim < 0) {
        std::cerr << "Unable to detect num_dim from node: " << TF_OperationName(oper) << std::endl;
        is_node_legal = false;
        return info;
    } else {
        shape_arr.resize(num_dim);
        TF_GraphGetTensorShape(this->model.graph,
                               info.op,
                               shape_arr.data(), num_dim,
                               this->model.status);
        if (TF_GetCode(this->model.status) != TF_OK) {
            std::cerr << TF_Message(this->model.status) << std::endl;
            is_node_legal = false;
            return info;
        }
        info.shape = shape_arr;
    }



    std::cout << "Tensor_shape:         ";
    print_shape(shape_arr);
    std::cout << "--------------------------------------\n" << std::endl;


    is_node_legal = true;
    return info;
}

// void NoOpDeallocator(void* data, size_t, void*) {}

// std::vector<tensorflow::int64> get_shape_from_tensor(tensorflow::Tensor& tensor) {
//     std::vector<tensorflow::int64> shape;
//     int64_t num_dimensions = tensor.shape().dims();
//     for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
//         shape.push_back(tensor.shape().dim_size(ii_dim));
//     }
//     return shape;
// }
