
#ifndef C_INTERFACE_H
#define C_INTERFACE_H


#include <string>
// #include "predictor.h"
#include "c_interface/c_config.h"

class Predictor;

extern "C" {

Predictor* createPredictor(const initialize_config& config);

void deletePredictor(Predictor* self);

void PredictorSetDataCount(Predictor* self, int data_count);

int64_t PredictorGetDataCount(Predictor* self);

void PredictorRegisterNode(Predictor* self, const char* node_name, Settings::NodeType type);

bool PredictorSetNodeData(Predictor* self, const char* node_name, void* p_data, int data_length, 
                          IODataType dtype, io_config config);

void PredictorRun(Predictor* self);

bool PredictorGetNodeData(Predictor* self, const char* node_name, void* p_data, int data_length, 
                          IODataType dtype, io_config config);

void PredictorPrintOperations(Predictor* self);

void PredictorPrintOperationName(Predictor* self, const char* node_name);

}


#endif // C_INTERFACE_H
