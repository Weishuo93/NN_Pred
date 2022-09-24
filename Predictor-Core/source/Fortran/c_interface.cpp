// #ifndef C_INTERFACE_H
// #define C_INTERFACE_H

#include "predictor.h"

// #ifdef __cplusplus
extern "C" {
// #endif

    Predictor* createPredictorFromPB (char* pb_file) {
        return new Predictor(pb_file);
    }

    Predictor* createPredictorFromSavedModel(char* model_dir, char* tags) {
        return new Predictor(model_dir, tags);
    }

    void deletePredictor(Predictor* self) {
        delete self;
    }

    void PredictorSetDataCount(Predictor* self, int data_count) {
        self->set_data_count(data_count);
    }

    void PredictorRegisterInputNode(Predictor* self, char* node_name) {
        self->regist_node(node_name, Predictor::INPUT_NODE);
    }

    void PredictorRegisterOutputNode(Predictor* self, char* node_name) {
        self->regist_node(node_name, Predictor::OUTPUT_NODE);
    }

    void PredictorSetNodeDataDouble(Predictor* self, char* node_name, double* p_data, int data_length) {
        self->set_node_data<double>(node_name, p_data, data_length);
    }

    void PredictorSetNodeDataFloat(Predictor* self, char* node_name, float* p_data, int data_length) {
        self->set_node_data<float>(node_name, p_data, data_length);
    }

    void PredictorSetNodeDataInt(Predictor* self, char* node_name, int* p_data, int data_length) {
        self->set_node_data<int>(node_name, p_data, data_length);
    }

    void PredictorRun(Predictor* self) {
        self->run();
    }

    void PredictorGetNodeDataDouble(Predictor * self, char* node_name, double* p_data, int data_length) {
        self->get_node_data<double>(node_name, p_data, data_length);
    }

    void PredictorGetNodeDataFloat(Predictor* self, char* node_name, float* p_data, int data_length) {
        self->get_node_data<float>(node_name, p_data, data_length);
    }

    void PredictorGetNodeDataInt(Predictor* self, char* node_name, int* p_data, int data_length) {
        self->get_node_data<int>(node_name, p_data, data_length);
    }

    




    void PredictorSetNodeDataTransposeDouble(Predictor* self, char* node_name, double* p_data, int data_length) {
        self->set_node_data<double>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

    void PredictorSetNodeDataTransposeFloat(Predictor* self, char* node_name, float* p_data, int data_length) {
        self->set_node_data<float>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

    void PredictorSetNodeDataTransposeInt(Predictor* self, char* node_name, int* p_data, int data_length) {
        self->set_node_data<int>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeDouble(Predictor * self, char* node_name, double* p_data, int data_length) {
        self->get_node_data<double>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeFloat(Predictor* self, char* node_name, float* p_data, int data_length) {
        self->get_node_data<float>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

    void PredictorGetNodeDataTransposeInt(Predictor* self, char* node_name, int* p_data, int data_length) {
        self->get_node_data<int>(node_name, p_data, data_length, Predictor::ColumnMajor);
    }

// #ifdef __cplusplus
}
// #endif

// #endif // C_INTERFACE_H
