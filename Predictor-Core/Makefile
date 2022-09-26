# Compiler 
# -------------------------------------------------------------------------------
CXX := g++
EIGEN_CXX_FLAG := -DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS -DEIGEN_MPL2_ONLY
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
# EIGEN_HOME := $(PWD)/third_party/eigen-3.3.9

# Core-predictor compile options
INCS := -I$(TF_HOME)/include -I./include
LIBS := -L$(TF_HOME)/lib -ltensorflow

# Local settings
OUTPUT_DIR := $(PWD)/outputs

# Outputs for core predictor library
CXX_SO   := $(OUTPUT_DIR)/lib/libPredictor.so

# Fortran extension settings
F90_SRC := source/Fortran
F90_INC := ./include
F90_SO  := $(OUTPUT_DIR)/lib/libPredictor_fortran.so


# Test Program
CXX_EXE := $(OUTPUT_DIR)/bin/testcxx
F90_EXE := $(OUTPUT_DIR)/bin/testf90



# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean cxxso f90so cxxtest f90test run runf guard-%

all: cxxso

run: cxxso cxxtest 
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(TF_HOME)/lib:$$LD_LIBRARY_PATH  $(CXX_EXE)

runf: cxxso f90so f90test 
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(TF_HOME)/lib:$$LD_LIBRARY_PATH  $(F90_EXE)

clean:
	rm -rf $(OUTPUT_DIR)/bin/*
	rm -rf $(OUTPUT_DIR)/lib/*
	rm -rf include/ml_predictor.mod


guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

cxxso: guard-TF_HOME guard-EIGEN_HOME $(CXX_SO)

f90so: cxxso $(F90_SO)

cxxtest: cxxso $(CXX_EXE)

f90test: cxxso f90so $(F90_EXE)





# Core C++ Predictor
# ++++++++++++++++++++
$(CXX_SO): source/C++/predictor_c.cpp 
	mkdir -p $(OUTPUT_DIR)/bin
	mkdir -p $(OUTPUT_DIR)/lib
	$(CXX) $(CXXFLAG) $(EIGEN_CXX_FLAG) $(INCS) $(EIGEN_INCS) -fPIC -shared $< -o $@  $(LIBS)


# Fortran Wrapper
# ++++++++++++++++++++

$(F90_SO): $(F90_SRC)/c_interface.cpp $(F90_SRC)/predictor_mod.f90
	$(F90) $(F90FLAG) -I $(F90_INC) -fPIC -shared $^ -o $@ -L$(OUTPUT_DIR)/lib/ -lstdc++ -J $(F90_INC)


# C++ Test Program
# ++++++++++++++++++++
$(CXX_EXE): test/test_AplusB.cpp 
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(TF_HOME)/lib:$$LD_LIBRARY_PATH $(CXX) -I./include  $< -o $@  -L$(OUTPUT_DIR)/lib/ -lPredictor 


# Fortran Test Program
# ++++++++++++++++++++
$(F90_EXE): test/fortran_test_AplusB.f90 
	LD_LIBRARY_PATH=$(OUTPUT_DIR)/lib:$(TF_HOME)/lib:$$LD_LIBRARY_PATH $(F90) $(F90FLAG) -I $(F90_INC)  $< -o $@  -L$(OUTPUT_DIR)/lib/ -lPredictor_fortran -lPredictor -lstdc++ 





