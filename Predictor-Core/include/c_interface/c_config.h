
#ifndef C_CONFIG_H
#define C_CONFIG_H


#include <string>
#include "settings.h"

// Following Definitions are for C extensions 
enum IODataType {
    FLOAT_TYPE,
    DOUBLE_TYPE,
    INT32_T_TYPE,
    INT16_T_TYPE,
    INT8_T_TYPE,
    UINT8_T_TYPE,
    UINT16_T_TYPE,
    UINT32_T_TYPE,
};


struct initialize_config {
    char* model_dir;
    char* tags;
    int intra_op_parallelism_threads = 1;
    int inter_op_parallelism_threads = 1;

    initialize_config() : model_dir(nullptr), tags(nullptr) {
        model_dir = new char[1024];
        tags = new char[1024];
        std::memset(model_dir, 0, 1024);
        std::memset(tags, 0, 1024);
    }

    ~initialize_config() {
        delete[] model_dir;
        delete[] tags;
    }

};

struct io_config {
        Settings::DataLayout layout = Settings::RowMajor;
        Settings::CopyMethod method = Settings::Eigen;
};


#endif // C_CONFIG_H
