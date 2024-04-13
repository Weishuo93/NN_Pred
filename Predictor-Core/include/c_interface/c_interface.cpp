#include "c_interface/c_interface.h"
#include "predictor.h"
#include <iostream>


extern "C" {

Predictor* createPredictor(const initialize_config& config) {
    if (config.tags[0] == '\0'){
        return new Predictor(config.model_dir, config.intra_op_parallelism_threads, config.inter_op_parallelism_threads);
    } else {
        return new Predictor(config.model_dir, config.tags, config.intra_op_parallelism_threads, config.inter_op_parallelism_threads);
    }
}

void deletePredictor(Predictor* self) {
    delete self;
}

void PredictorSetDataCount(Predictor* self, int data_count) {
    self->set_data_count(data_count);
}

int64_t PredictorGetDataCount(Predictor* self) {
    return self->get_data_count();
}

void PredictorRegisterNode(Predictor* self, const char* node_name, Settings::NodeType type) {
    self->regist_node(node_name, type);
}

bool PredictorSetNodeData(Predictor* self, const char* node_name, void* p_data, int data_length, 
                            IODataType dtype, io_config config) {
    switch (dtype) {
        case FLOAT_TYPE:
            return self->set_node_data(node_name, static_cast<float*>(p_data), data_length, config.layout, config.method);
            break;
        case DOUBLE_TYPE:
            return self->set_node_data(node_name, static_cast<double*>(p_data), data_length, config.layout, config.method);
            break;
        case INT32_T_TYPE:
            return self->set_node_data(node_name, static_cast<int32_t*>(p_data), data_length, config.layout, config.method);
            break;
        case INT16_T_TYPE:
            return self->set_node_data(node_name, static_cast<int16_t*>(p_data), data_length, config.layout, config.method);
            break;
        case INT8_T_TYPE:
            return self->set_node_data(node_name, static_cast<int8_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT8_T_TYPE:
            return self->set_node_data(node_name, static_cast<uint8_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT16_T_TYPE:
            return self->set_node_data(node_name, static_cast<uint16_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT32_T_TYPE:
            return self->set_node_data(node_name, static_cast<uint32_t*>(p_data), data_length, config.layout, config.method);
            break;
    }
    std::cerr << "Unsupported data type in C_interface.";
    return false;
}

void PredictorRun(Predictor* self) {
    self->run();
}

bool PredictorGetNodeData(Predictor* self, const char* node_name, void* p_data, int data_length, 
                            IODataType dtype, io_config config) {
    switch (dtype) {
        case FLOAT_TYPE:
            return self->get_node_data(node_name, static_cast<float*>(p_data), data_length, config.layout, config.method);
            break;
        case DOUBLE_TYPE:
            return self->get_node_data(node_name, static_cast<double*>(p_data), data_length, config.layout, config.method);
            break;
        case INT32_T_TYPE:
            return self->get_node_data(node_name, static_cast<int32_t*>(p_data), data_length, config.layout, config.method);
            break;
        case INT16_T_TYPE:
            return self->get_node_data(node_name, static_cast<int16_t*>(p_data), data_length, config.layout, config.method);
            break;
        case INT8_T_TYPE:
            return self->get_node_data(node_name, static_cast<int8_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT8_T_TYPE:
            return self->get_node_data(node_name, static_cast<uint8_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT16_T_TYPE:
            return self->get_node_data(node_name, static_cast<uint16_t*>(p_data), data_length, config.layout, config.method);
            break;
        case UINT32_T_TYPE:
            return self->get_node_data(node_name, static_cast<uint32_t*>(p_data), data_length, config.layout, config.method);
            break;
    }
    std::cerr << "Unsupported data type in C_interface.";
    return false;
}


void PredictorPrintOperations(Predictor* self){
    self->print_operations();
}

void PredictorPrintOperationName(Predictor* self, const char* node_name) {
    self->print_operations(node_name);
}

}