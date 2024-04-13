#ifndef DYN_PREDICTOR_H
#define DYN_PREDICTOR_H

#include <vector>
#include <string>
#include "settings.h"

class DynPredictorImpl;

class DynPredictor {

   public:

    explicit DynPredictor(std::string pbfile);
    explicit DynPredictor(std::string folder, std::string tag);

    explicit DynPredictor(std::string pbfile, int intra_op_parallelism_threads, int inter_op_parallelism_threads);
    explicit DynPredictor(std::string folder, std::string tag, int intra_op_parallelism_threads, int inter_op_parallelism_threads);

    virtual ~DynPredictor();

    void print_operations();
    void print_operations(std::string node_name);

    void regist_node(std::string name, Settings::NodeType type);

    void set_data_count(int64_t cnt);
    int64_t get_data_count();

    void run();

    // Data Access functions:
    // Set and get data using std::vector
    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data);

    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout);

    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout, Settings::CopyMethod method);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data, Settings::DataLayout layout, Settings::CopyMethod method);

    // Set and Get data function using raw data pointer

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size);

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout);

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size, Settings::DataLayout layout, Settings::CopyMethod method);

   private:
    DynPredictorImpl* d;
};

#endif // DYN_PREDICTOR_H
