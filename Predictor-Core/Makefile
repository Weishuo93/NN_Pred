# Compiler 
# -------------------------------------------------------------------------------
CXX := g++
EIGEN_CXX_FLAG := -DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS -DEIGEN_MPL2_ONLY
ONNX_CXX_FLAG := 
CXXFLAG := -O3 -D_CPU_SERIAL

# CXXFLAG := -O3  # Default
# CXXFLAG := -O3 -D_CPU_SERIAL  # Each predictor use single core
# CXXFLAG := -g -D_DEBUG # Debug Mode


F90 := gfortran
F90FLAG := -O3
# F90FLAG := -O3

# Tensorflow Installation
# -------------------------------------------------------------------------------
TF_HOME  := $(MY_TF_HOME)
# TF_HOME  := $(PWD)/third_party/TF_libs

# Eigen Installation
# -------------------------------------------------------------------------------
EIGEN_HOME := $(MY_EIGEN_HOME)
EIGEN_INCS := -I$(EIGEN_HOME)

# Onnx Installation
# -------------------------------------------------------------------------------
ONNX_HOME := $(MY_ONNX_HOME)


# Core-predictor compile options
INCS := -I./include

TF_INCS := -I$(TF_HOME)/include 
TF_LIBS := -L$(TF_HOME)/lib -ltensorflow

ONNX_INCS := -I$(ONNX_HOME)/include 
ONNX_LIBS := -L$(ONNX_HOME)/lib -lonnxruntime

# Local settings
OUTPUT_DIR := $(PWD)/outputs

# Outputs for core predictor library
CXX_SO_TF   := $(OUTPUT_DIR)/lib/libPredictor_tf.so
CXX_SO_ONNX := $(OUTPUT_DIR)/lib/libPredictor_onnx.so
CXX_SO :=$(OUTPUT_DIR)/lib/libPredictor.so

# Fortran extension settings
F90_SRC := source/Fortran
F90_INC := ./include
F90_SO  := $(OUTPUT_DIR)/lib/libPredictor_fortran.so


# Test Program
CXX_EXE := $(OUTPUT_DIR)/bin/testcxx
F90_EXE := $(OUTPUT_DIR)/bin/testf90

# Test Model Path:
MODEL_PATH_ONNX :=$(PWD)/test/models/AplusB.onnx
MODEL_PATH_TF :=$(PWD)/test/models/simple_graph_tf2.pb


# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean cxxso f90so cxxtest f90test run runf guard-%

all: cxxso

run: cxxso cxxtest guard-NNPRED_BACKEND
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(${NNPRED_BACKEND}_HOME)/lib:$$LD_LIBRARY_PATH  $(CXX_EXE) $(MODEL_PATH_${NNPRED_BACKEND})

runf: cxxso f90so f90test 
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(${NNPRED_BACKEND}_HOME)/lib:$$LD_LIBRARY_PATH  $(F90_EXE) $(MODEL_PATH_${NNPRED_BACKEND})

clean:
	rm -rf $(OUTPUT_DIR)/bin/*
	rm -rf $(OUTPUT_DIR)/lib/*
	rm -rf include/ml_predictor.mod


guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

alias-predictor: guard-NNPRED_BACKEND
	@ if [ "${NNPRED_BACKEND}" = "ONNX" ]; then \
		echo "Using ${NNPRED_BACKEND} Backends"; \
		ln -sf $(CXX_SO_${NNPRED_BACKEND}) $(CXX_SO); \
	elif [ "${NNPRED_BACKEND}" = "TF" ]; then \
		echo "Using ${NNPRED_BACKEND} Backends"; \
		ln -sf $(CXX_SO_${NNPRED_BACKEND}) $(CXX_SO); \
	fi

cxxso_tf: guard-TF_HOME guard-EIGEN_HOME $(CXX_SO_TF)

cxxso_onnx: guard-ONNX_HOME guard-EIGEN_HOME $(CXX_SO_ONNX)

cxxso: cxxso_onnx cxxso_tf alias-predictor

f90so: cxxso $(F90_SO)

cxxtest: cxxso $(CXX_EXE)

f90test: cxxso f90so $(F90_EXE)





# Core C++ Predictor
# ++++++++++++++++++++
$(CXX_SO_TF): source/C++/predictor_tf.cpp 
	mkdir -p $(OUTPUT_DIR)/bin
	mkdir -p $(OUTPUT_DIR)/lib
	$(CXX) $(CXXFLAG) $(EIGEN_CXX_FLAG) $(INCS) $(TF_INCS) $(EIGEN_INCS) -fPIC -shared $< -o $@  $(TF_LIBS)


$(CXX_SO_ONNX): source/C++/predictor_onnx.cpp 
	mkdir -p $(OUTPUT_DIR)/bin
	mkdir -p $(OUTPUT_DIR)/lib
	$(CXX) $(CXXFLAG) $(EIGEN_CXX_FLAG) $(INCS) $(ONNX_INCS) $(EIGEN_INCS) -fPIC -shared $< -o $@  $(ONNX_LIBS)


# Fortran Wrapper
# ++++++++++++++++++++
$(F90_SO): $(F90_SRC)/c_interface.cpp $(F90_SRC)/predictor_mod.f90
	$(F90) $(F90FLAG) -I $(F90_INC) -fPIC -shared $^ -o $@ -L$(OUTPUT_DIR)/lib/ -lstdc++ -J $(F90_INC)


# C++ Test Program
# ++++++++++++++++++++
$(CXX_EXE): test/test_AplusB.cpp 
	$(CXX) -I./include  $< -o $@  -L$(OUTPUT_DIR)/lib/ -L$(${NNPRED_BACKEND}_HOME)/lib -lPredictor 


# Fortran Test Program
# ++++++++++++++++++++
$(F90_EXE): test/fortran_test_AplusB.f90 
	$(F90) $(F90FLAG) -I $(F90_INC)  $< -o $@  -L$(OUTPUT_DIR)/lib/ -L$(${NNPRED_BACKEND}_HOME)/lib -lPredictor_fortran -lPredictor -lstdc++ 


# $(CXX_SO):
# 	ln -s $(CXX_SO_${NNPRED_BACKEND}) $(CXX_SO)





