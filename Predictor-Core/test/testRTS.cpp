#include <iostream>

// Header files
#include <vector>       // C++ standard header
#include <string>       // C++ standard header
#include "dyn_predictor.h"  // Predictor header


int main(int argc, char const *argv[]) {

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    std::string model_path(argv[1]);
    // Load Model:
    // Predictor pd("AplusB.pb");  // Model's path or filename
    DynPredictor pd(model_path);

    // Register node:
    // Inputs:
    // Settings::INPUT_NODE is the node type enumerate
    pd.regist_node("input_a", Settings::INPUT_NODE);
    pd.regist_node("input_b", Settings::INPUT_NODE);
    // Outputs:
    // Settings::OUTPUT_NODE is the node type enumerate
    pd.regist_node("result", Settings::OUTPUT_NODE);

    // Set the number of data instances (n=5)
    pd.set_data_count(3);

    //End of initialization

    // Create external source of input/output data array:
    // Inputs:
    std::vector<float> vec_input1 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    std::vector<int> vec_input2 = {6, 5, 4, 3, 2, 1};
    // Outputs:
    std::vector<double> vec_out(6);

    // Set data for input nodes
    pd.set_node_data("input_a", vec_input1);
    pd.set_node_data("input_b", vec_input2);

    // Run model 
    pd.run();

    // Get output into the target container
    pd.get_node_data("result", vec_out, Settings::ColumnMajor, Settings::Simple);

    // Check results, expected calculation results:
    // [7.1, 7.2, 7.3, 7.4, 7.5, 7.6]
    std::cout << vec_out[0] << ", " << vec_out[1] << ", " << vec_out[2] << ", " 
              << vec_out[3] << ", " << vec_out[4] << ", " << vec_out[5] << std::endl;

    return 0;
}