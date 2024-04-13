#ifndef PREDICTOR_SETTING_H
#define PREDICTOR_SETTING_H

#include <map>

namespace Settings {
    
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


};

#endif // PREDICTOR_SETTING_H
