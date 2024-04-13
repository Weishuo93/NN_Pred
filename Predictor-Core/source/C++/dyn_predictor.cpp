#include <cstdlib> // 用于 getenv
#include <cstring>
#include <iostream>
#include <dlfcn.h> // 动态加载库所需头文件
#include "dyn_predictor.h" // 假设这是您的API头文件
#include "c_interface/c_interface.h" // 假设这是您的API头文件



std::string get_library_name() {
    const char* env = std::getenv("NNPRED_BACKEND");
    std::string predictorType = env ? env : "";

    // 检查predictorType是否为"TF"或"ONNX"
    if (predictorType == "TF") {
        return "libPredictor_tf.so";
    } else if (predictorType == "ONNX") {
        return "libPredictor_onnx.so";
    } else {
        // 如果环境变量不存在或值不是"TF"或"ONNX"
        if (predictorType.empty()) {
            std::cout << "$NNPRED_BACKEND not set, using default TF." << std::endl;
        } else {
            std::cout << "Unsupported backend: " << predictorType << ", using default TF.." << std::endl;
        }
        // 默认返回TF的库名
        return "libPredictor_tf.so";
    }
}





class DynamicLibLoader {
public:

    using createPredictorFunc = Predictor* (*) (const initialize_config&);
    using deletePredictorFunc = void (*) (Predictor*);
    using PredictorSetDataCountFunc = void (*) (Predictor* , int);
    using PredictorGetDataCountFunc = int64_t (*) (Predictor*) ;
    using PredictorRegisterNodeFunc = void (*) (Predictor*, const char*, Settings::NodeType);
    using PredictorSetNodeDataFunc = bool (*) (Predictor*, const char*, void*, int, IODataType, io_config);
    using PredictorRunFunc = void (*) (Predictor*);
    using PredictorGetNodeDataFunc = bool (*) (Predictor*, const char*, void*, int, IODataType, io_config);
    using PredictorPrintOperationsFunc = void (*) (Predictor*);
    using PredictorPrintOperationNameFunc = void (*) (Predictor*, const char*);

    // 存储函数指针的成员变量
    createPredictorFunc createPredictor = nullptr;
    deletePredictorFunc deletePredictor = nullptr;
    PredictorSetDataCountFunc PredictorSetDataCount = nullptr;
    PredictorGetDataCountFunc PredictorGetDataCount = nullptr;
    PredictorRegisterNodeFunc PredictorRegisterNode = nullptr;
    PredictorSetNodeDataFunc PredictorSetNodeData = nullptr;
    PredictorRunFunc PredictorRun = nullptr;
    PredictorGetNodeDataFunc PredictorGetNodeData = nullptr;
    PredictorPrintOperationsFunc PredictorPrintOperations = nullptr;
    PredictorPrintOperationNameFunc PredictorPrintOperationName = nullptr;


    // 构造函数：加载共享库
    DynamicLibLoader(const std::string& libPath) {
        loadLibrary(libPath);
    }

    // 析构函数：关闭共享库
    ~DynamicLibLoader() {
        if (handle) {
            dlclose(handle);
        }
    }

        // 禁止拷贝和赋值
    DynamicLibLoader(const DynamicLibLoader&) = delete;
    DynamicLibLoader& operator=(const DynamicLibLoader&) = delete;

private:
    void* handle = nullptr; // 用于存储共享库的句柄

    void loadLibrary(const std::string& libPath) {
        handle = dlopen(libPath.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Cannot open library: " << dlerror() << '\n';
            return;
        }

        // 加载每个函数并赋值给成员变量
        *(void**)(&createPredictor) = dlsym(handle, "createPredictor");
        *(void**)(&deletePredictor) = dlsym(handle, "deletePredictor");
        *(void**)(&PredictorSetDataCount) = dlsym(handle, "PredictorSetDataCount");
        *(void**)(&PredictorGetDataCount) = dlsym(handle, "PredictorGetDataCount");
        *(void**)(&PredictorRegisterNode) = dlsym(handle, "PredictorRegisterNode");
        *(void**)(&PredictorSetNodeData) = dlsym(handle, "PredictorSetNodeData");
        *(void**)(&PredictorRun) = dlsym(handle, "PredictorRun");
        *(void**)(&PredictorGetNodeData) = dlsym(handle, "PredictorGetNodeData");
        *(void**)(&PredictorPrintOperations) = dlsym(handle, "PredictorPrintOperations");
        *(void**)(&PredictorPrintOperationName) = dlsym(handle, "PredictorPrintOperationName");


        // 检查错误
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "Loading symbols failed: " << dlsym_error << '\n';
            dlclose(handle);
            handle = nullptr;
        }
    }
};


class DynPredictorImpl {
public:
    Predictor* pd;
    DynamicLibLoader* loader;
    initialize_config int_conf;
    io_config input_conf;
    io_config output_conf;

    DynPredictorImpl(std::string model_dir, 
                     int intra_para, int inter_para, 
                     std::string tags="", std::string lib_name = "")
                     : int_conf(), input_conf(), output_conf() {

        if (lib_name.empty()){
            lib_name = get_library_name();
        }
        
        loader = new DynamicLibLoader(lib_name);

        std::strncpy(int_conf.model_dir, model_dir.c_str(), model_dir.length() + 1);
        int_conf.intra_op_parallelism_threads = intra_para;
        int_conf.inter_op_parallelism_threads = inter_para;
        
        if (!tags.empty()){
            std::strncpy(int_conf.tags, tags.c_str(), tags.length() + 1);
        } else {
            int_conf.tags[0] == '\0';
        }

        pd = loader->createPredictor(int_conf);
    }

    ~DynPredictorImpl(){
        loader->deletePredictor(pd);
        delete loader;
    }

};


DynPredictor::DynPredictor(std::string pbfile, int intra_op_parallelism_threads, int inter_op_parallelism_threads)
    :d(nullptr) {
    d = new DynPredictorImpl(pbfile, intra_op_parallelism_threads, inter_op_parallelism_threads);
}

DynPredictor::DynPredictor(std::string folder, std::string tag, int intra_op_parallelism_threads, int inter_op_parallelism_threads)
    :d(nullptr) {
    d = new DynPredictorImpl(folder, intra_op_parallelism_threads, inter_op_parallelism_threads, tag);
}

#ifdef _CPU_SERIAL
DynPredictor::DynPredictor(std::string pbfile) : DynPredictor(pbfile, 1, 1) {}
DynPredictor::DynPredictor(std::string folder, std::string tag) : DynPredictor(folder, tag, 1, 1) {}
#else
DynPredictor::DynPredictor(std::string pbfile) : DynPredictor(pbfile, 0, 0) {}
DynPredictor::DynPredictor(std::string folder, std::string tag) : DynPredictor(folder, tag, 0, 0) {}
#endif


DynPredictor::~DynPredictor(){
    delete d;
}

void DynPredictor::print_operations(){
    d->loader->PredictorPrintOperations(d->pd);
}


void DynPredictor::print_operations(std::string node_name) {
    d->loader->PredictorPrintOperationName(d->pd, node_name.c_str());
}

void DynPredictor::regist_node(std::string name, Settings::NodeType type) {
    d->loader->PredictorRegisterNode(d->pd, name.c_str(), type);
}



void DynPredictor::set_data_count(int64_t cnt){
    d->loader->PredictorSetDataCount(d->pd, cnt);
}

int64_t DynPredictor::get_data_count() {
    return d->loader->PredictorGetDataCount(d->pd);
}


void DynPredictor::run() {
    d->loader->PredictorRun(d->pd);
}


template <typename T>
IODataType deduce_type() {
    if (std::is_same<T, float>::value)
        return IODataType::FLOAT_TYPE;
    if (std::is_same<T, double>::value)
        return IODataType::DOUBLE_TYPE;
    if (std::is_same<T, int32_t>::value)
        return IODataType::INT32_T_TYPE;
    if (std::is_same<T, int16_t>::value)
        return IODataType::INT16_T_TYPE;
    if (std::is_same<T, int8_t>::value)
        return IODataType::INT8_T_TYPE;
    if (std::is_same<T, uint8_t>::value)
        return IODataType::UINT8_T_TYPE;
    if (std::is_same<T, uint16_t>::value)
        return IODataType::UINT16_T_TYPE;
    if (std::is_same<T, uint32_t>::value)
        return IODataType::UINT32_T_TYPE;
    std::cerr << "Currently not supported data type." << std::endl;
    return static_cast<IODataType>(-100);
}

// instantiate template functions
template IODataType deduce_type<float>();
template IODataType deduce_type<double>();
template IODataType deduce_type<int8_t>();
template IODataType deduce_type<int16_t>();
template IODataType deduce_type<int32_t>();
template IODataType deduce_type<int64_t>();
template IODataType deduce_type<uint8_t>();
template IODataType deduce_type<uint16_t>();
template IODataType deduce_type<uint32_t>();


// enum IODataType {
//     FLOAT_TYPE,
//     DOUBLE_TYPE,
//     INT32_T_TYPE,
//     INT16_T_TYPE,
//     INT8_T_TYPE,
//     UINT8_T_TYPE,
//     UINT16_T_TYPE,
//     UINT32_T_TYPE,
// };


// Data Access functions:
// Set and get data using std::vector

template <typename T>
bool DynPredictor::set_node_data(std::string name, T* p_data, int array_size, 
                                 Settings::DataLayout layout, Settings::CopyMethod method) {
    IODataType T_type = deduce_type<T>();
    d->input_conf.layout = layout;
    d->input_conf.method = method;
    return d->loader->PredictorSetNodeData(d->pd, name.c_str(), static_cast<void*>(p_data), array_size, T_type, d->input_conf);
}

template bool DynPredictor::set_node_data<float>(std::string name, float* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<double>(std::string name, double* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);

template <typename T>
bool DynPredictor::set_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout){
    return this->set_node_data(name, p_data, array_size, layout, Settings::Eigen);
}

template bool DynPredictor::set_node_data<float>(std::string name, float* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<double>(std::string name, double* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Settings::DataLayout layout);

template <typename T>
bool DynPredictor::set_node_data(std::string name, T* p_data, int array_size) {
    return this->set_node_data(name, p_data, array_size, Settings::RowMajor, Settings::Eigen);
}

template bool DynPredictor::set_node_data<float>(std::string name, float* p_data, int array_size);
template bool DynPredictor::set_node_data<double>(std::string name, double* p_data, int array_size);
template bool DynPredictor::set_node_data<int32_t>(std::string name, int32_t* p_data, int array_size);
template bool DynPredictor::set_node_data<int16_t>(std::string name, int16_t* p_data, int array_size);
template bool DynPredictor::set_node_data<int8_t>(std::string name, int8_t* p_data, int array_size);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size);

// =====================================================================================================================
template <typename T>
bool DynPredictor::set_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout, Settings::CopyMethod method){
    return this->set_node_data(name, data.data(), data.size(), layout, method);
}

template bool DynPredictor::set_node_data<float>(std::string name, std::vector<float>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<double>(std::string name, std::vector<double>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);



template <typename T>
bool DynPredictor::set_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout) {
    return this->set_node_data(name, data, layout, Settings::Eigen);
}

template bool DynPredictor::set_node_data<float>(std::string name, std::vector<float>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<double>(std::string name, std::vector<double>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Settings::DataLayout layout);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Settings::DataLayout layout);

template <typename T>
bool DynPredictor::set_node_data(std::string name, std::vector<T>& data){
    return this->set_node_data(name, data, Settings::RowMajor, Settings::Eigen);
}

template bool DynPredictor::set_node_data<float>(std::string name, std::vector<float>& data);
template bool DynPredictor::set_node_data<double>(std::string name, std::vector<double>& data);
template bool DynPredictor::set_node_data<int32_t>(std::string name, std::vector<int32_t>& data);
template bool DynPredictor::set_node_data<int16_t>(std::string name, std::vector<int16_t>& data);
template bool DynPredictor::set_node_data<int8_t>(std::string name, std::vector<int8_t>& data);
template bool DynPredictor::set_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data);
template bool DynPredictor::set_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data);
template bool DynPredictor::set_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data);

// ==================================================================================================================================================

// ====================================================================================================================================================

template <typename T>
bool DynPredictor::get_node_data(std::string name, T* p_data, int array_size, 
                                 Settings::DataLayout layout, Settings::CopyMethod method) {
    IODataType T_type = deduce_type<T>();
    d->output_conf.layout = layout;
    d->output_conf.method = method;
    return d->loader->PredictorGetNodeData(d->pd, name.c_str(), static_cast<void*>(p_data), array_size, T_type, d->output_conf);
}

template bool DynPredictor::get_node_data<float>(std::string name, float* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<double>(std::string name, double* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);

template <typename T>
bool DynPredictor::get_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout){
    return this->get_node_data(name, p_data, array_size, layout, Settings::Eigen);
}

template bool DynPredictor::get_node_data<float>(std::string name, float* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<double>(std::string name, double* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size, Settings::DataLayout layout);

template <typename T>
bool DynPredictor::get_node_data(std::string name, T* p_data, int array_size) {
    return this->get_node_data(name, p_data, array_size, Settings::RowMajor, Settings::Eigen);
}

template bool DynPredictor::get_node_data<float>(std::string name, float* p_data, int array_size);
template bool DynPredictor::get_node_data<double>(std::string name, double* p_data, int array_size);
template bool DynPredictor::get_node_data<int32_t>(std::string name, int32_t* p_data, int array_size);
template bool DynPredictor::get_node_data<int16_t>(std::string name, int16_t* p_data, int array_size);
template bool DynPredictor::get_node_data<int8_t>(std::string name, int8_t* p_data, int array_size);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, uint8_t* p_data, int array_size);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, uint16_t* p_data, int array_size);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, uint32_t* p_data, int array_size);

// =====================================================================================================================
template <typename T>
bool  DynPredictor::get_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout, Settings::CopyMethod method){
    return get_node_data(name, data.data(), data.size(), layout, method);
}

template bool DynPredictor::get_node_data<float>(std::string name, std::vector<float>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<double>(std::string name, std::vector<double>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Settings::DataLayout layout, Settings::CopyMethod method);



template <typename T>
bool DynPredictor::get_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout) {
    return this->get_node_data(name, data, layout, Settings::Eigen);
}

template bool DynPredictor::get_node_data<float>(std::string name, std::vector<float>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<double>(std::string name, std::vector<double>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data, Settings::DataLayout layout);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data, Settings::DataLayout layout);

template <typename T>
bool DynPredictor::get_node_data(std::string name, std::vector<T>& data){
    return this->get_node_data(name, data, Settings::RowMajor, Settings::Eigen);
}

template bool DynPredictor::get_node_data<float>(std::string name, std::vector<float>& data);
template bool DynPredictor::get_node_data<double>(std::string name, std::vector<double>& data);
template bool DynPredictor::get_node_data<int32_t>(std::string name, std::vector<int32_t>& data);
template bool DynPredictor::get_node_data<int16_t>(std::string name, std::vector<int16_t>& data);
template bool DynPredictor::get_node_data<int8_t>(std::string name, std::vector<int8_t>& data);
template bool DynPredictor::get_node_data<uint8_t>(std::string name, std::vector<uint8_t>& data);
template bool DynPredictor::get_node_data<uint16_t>(std::string name, std::vector<uint16_t>& data);
template bool DynPredictor::get_node_data<uint32_t>(std::string name, std::vector<uint32_t>& data);

