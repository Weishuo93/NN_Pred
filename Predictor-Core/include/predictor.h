#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <vector>
#include <map>
#include <string>

class PredictorImpl;

class Predictor {
   public:
    enum NodeType {
        INPUT_NODE,
        OUTPUT_NODE,
    };

    enum DataLayout {
        RowMajor,
        ColumnMajor,
    };

    enum CopyMethod {
        Eigen,
        Simple,
    };

    explicit Predictor(std::string pbfile);
    explicit Predictor(std::string folder, std::string tag);

    explicit Predictor(std::string pbfile, uint8_t intra_op_parallelism_threads, uint8_t inter_op_parallelism_threads);
    explicit Predictor(std::string folder, std::string tag, uint8_t intra_op_parallelism_threads, uint8_t inter_op_parallelism_threads);

    virtual ~Predictor();

    void print_operations();
    void print_operations(std::string node_name);

    void regist_node(std::string name, NodeType type);

    void set_data_count(int cnt);
    int get_data_count();

    void run();

    // Data Access functions:
    // Set and get data using std::vector
    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data);

    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data, DataLayout layout);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data, DataLayout layout);

    template <typename T>
    bool set_node_data(std::string name, std::vector<T>& data, DataLayout layout, CopyMethod method);

    template <typename T>
    bool get_node_data(std::string name, std::vector<T>& data, DataLayout layout, CopyMethod method);

    // Set and Get data function using raw data pointer

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size);

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size, DataLayout layout);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size, DataLayout layout);

    template <typename T>
    bool set_node_data(std::string name, T* p_data, int array_size, DataLayout layout, CopyMethod method);

    template <typename T>
    bool get_node_data(std::string name, T* p_data, int array_size, DataLayout layout, CopyMethod method);

    // // Default Row major set and get data function 2D

    // template <typename T>
    // bool set_node_data(std::string name, std::vector<std::vector<T>>& data);

    // template <typename T>
    // bool get_node_data(std::string name, std::vector<std::vector<T>>& data);

    // // Set and Get data function with special layout requirements 2D

    // template <typename T>
    // bool set_node_data(std::string name, std::vector<std::vector<T>>& data, DataLayout layout);

    // template <typename T>
    // bool get_node_data(std::string name, std::vector<std::vector<T>>& data, DataLayout layout);

   private:
    PredictorImpl* d;
};

#endif // PREDICTOR_H
