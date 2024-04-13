// #ifndef C_FORTRAN_INTERFACE_H
// #define C_FORTRAN_INTERFACE_H

#include "dyn_predictor.h"

// #ifdef __cplusplus
extern "C" {
// #endif

    DynPredictor* createPredictorFromPB (char* pb_file) {
        return new DynPredictor(pb_file);
    }

    DynPredictor* createPredictorFromSavedModel(char* model_dir, char* tags) {
        return new DynPredictor(model_dir, tags);
    }

    void deletePredictor(DynPredictor* self) {
        delete self;
    }

    void PredictorSetDataCount(DynPredictor* self, int data_count) {
        self->set_data_count(data_count);
    }

    void PredictorRegisterInputNode(DynPredictor* self, char* node_name) {
        self->regist_node(node_name, Settings::INPUT_NODE);
    }

    void PredictorRegisterOutputNode(DynPredictor* self, char* node_name) {
        self->regist_node(node_name, Settings::OUTPUT_NODE);
    }

    void PredictorSetNodeDataDouble(DynPredictor* self, char* node_name, double* p_data, int data_length) {
        self->set_node_data<double>(node_name, p_data, data_length);
    }

    void PredictorSetNodeDataFloat(DynPredictor* self, char* node_name, float* p_data, int data_length) {
        self->set_node_data<float>(node_name, p_data, data_length);
    }

    void PredictorSetNodeDataInt(DynPredictor* self, char* node_name, int* p_data, int data_length) {
        self->set_node_data<int>(node_name, p_data, data_length);
    }

    void PredictorRun(DynPredictor* self) {
        self->run();
    }

    void PredictorGetNodeDataDouble(DynPredictor* self, char* node_name, double* p_data, int data_length) {
        self->get_node_data<double>(node_name, p_data, data_length);
    }

    void PredictorGetNodeDataFloat(DynPredictor* self, char* node_name, float* p_data, int data_length) {
        self->get_node_data<float>(node_name, p_data, data_length);
    }

    void PredictorGetNodeDataInt(DynPredictor* self, char* node_name, int* p_data, int data_length) {
        self->get_node_data<int>(node_name, p_data, data_length);
    }

    




    void PredictorSetNodeDataTransposeDouble(DynPredictor* self, char* node_name, double* p_data, int data_length) {
        self->set_node_data<double>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

    void PredictorSetNodeDataTransposeFloat(DynPredictor* self, char* node_name, float* p_data, int data_length) {
        self->set_node_data<float>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

    void PredictorSetNodeDataTransposeInt(DynPredictor* self, char* node_name, int* p_data, int data_length) {
        self->set_node_data<int>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeDouble(DynPredictor* self, char* node_name, double* p_data, int data_length) {
        self->get_node_data<double>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeFloat(DynPredictor* self, char* node_name, float* p_data, int data_length) {
        self->get_node_data<float>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeInt(DynPredictor* self, char* node_name, int* p_data, int data_length) {
        self->get_node_data<int>(node_name, p_data, data_length, Settings::ColumnMajor);
    }

// #ifdef __cplusplus
}
// #endif

// #endif // C_FORTRAN_INTERFACE_H
