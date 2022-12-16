

// file: NNPred.h:
class NNPredImpl;
// Do we need this? 
class NNPredImpl_TF;
class NNPredImpl_ONNX;
// 
class NNPred {
private:
    NNPredImpl* d;
public:
    explicit NNPred(char* model_path) {
        const char* backend_string = std::getenv("NNPRED_BACKEND");
        if (backend_string == "TF") {
            d = new NNPredImpl_TF(model_path);
        } else if (backend_string == "ONNX") {
            d = new NNPredImpl_ONNX(model_path);
        }

        
        

    }
    ~NNPred();

// Methods:
    set_data(...) {
        d->set_data();
    }

};


// file: NNPredImpl.h:
class NNPredImpl {
private:
    NNPredImpl
public:
    virtual NNPredImpl(/* args */);
    virtual ~NNPred();

    // Methods:
    virtual set_data 
    ...

};

// file: NNPredImpl_TF.cpp:
#include "NNPredImpl.h"
#include "TF_CAPI"
class NNPredImpl_TF: public NNPredImpl{
public:
    NNPredImpl_TF(/* args */);
    ~NNPredImpl_TF();
};

NNPredImpl_TF::NNPredImpl_TF(/* args */) {
}

NNPredImpl_TF::~NNPredImpl_TF() {
}

// file: NNPredImpl_ONNX.cpp:
#include "NNPredImpl.h"
#include "ONNX_CAPI"
class NNPredImpl_ONNX: public NNPredImpl{
public:
    NNPredImpl_ONNX(/* args */);
    ~NNPredImpl_ONNX();
};

NNPredImpl_ONNX::NNPredImpl_ONNX(/* args */) {
}

NNPredImpl_ONNX::~NNPredImpl_ONNX() {
}




