CXX := g++
# python3-config --cflags 
# to determine INCFLAG for compiling python
INCFLAG := -I/mnt/e/Linux_space/VirtualEnv/TF2/include/python3.6m -I/mnt/e/Linux_space/VirtualEnv/TF2/lib64/python3.6/site-packages/numpy/core/include -Wno-unused-result -Wsign-compare -DNDEBUG -DOPENSSL_LOAD_CONF -fwrapv -D_FORTIFY_SOURCE=2 -fmessage-length=0 -grecord-gcc-switches -O2 -Wall -D_FORTIFY_SOURCE=2 -fstack-protector-strong -funwind-tables -fasynchronous-unwind-tables -fstack-clash-protection -g

# python3-config --ldflags 
# to determine LINKFLAG for compiling python
LINKFLAG := -lpython3.6m -lpthread -ldl -lutil -lm -Xlinker -export-dynamic

# Test Program
CXX_EXE_OFF := ./testpy_off

CXX_EXE_SELF := ./testpy_self

.PHONY: all clean cxxexe_off cxxexe_self run

all: cxxexe_off cxxexe_self

run_off: cxxexe_off 
	$(CXX_EXE_OFF) multiply multiply 3 2

run_self: cxxexe_self 
	$(CXX_EXE_SELF) 

clean:
	rm -rf $(CXX_EXE_OFF)
	rm -rf $(CXX_EXE_SELF)
	

cxxexe_off: $(CXX_EXE_OFF)

cxxexe_self: $(CXX_EXE_SELF)

$(CXX_EXE_OFF): testPy.cpp 
	$(CXX) $(INCFLAG)  $< -o $@ $(LINKFLAG)

$(CXX_EXE_SELF): testPy_self.cpp 
	$(CXX) $(INCFLAG)  $< -o $@ $(LINKFLAG)