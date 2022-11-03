# OpenFOAM extension settings
OF_SRC := TF_OF_Predictor/lib
OF_TEST := TF_OF_Predictor/test


OF_SO  := $(FOAM_USER_LIBBIN)/libTF_OF_Predictor.so
OF_EXE := $(FOAM_USER_APPBIN)/TF_OF_testAplutB




# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean ofso oftest runof guard-%

all: ofso

ofso: guard-FOAM_USER_APPBIN guard-FOAM_USER_LIBBIN check-wmake $(OF_SO)

oftest: ofso $(OF_EXE)

runof: ofso oftest 
	cd $(OF_TEST) && $(OF_EXE)


clean:
	rm -rf $(FOAM_USER_LIBBIN)/libTF_OF_Predictor.so
	rm -rf $(FOAM_USER_APPBIN)/TF_OF_testAplusB
	wclean $(OF_SRC)
	wclean $(OF_TEST)

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

check-wmake:
ifeq (, $(shell which wmake))
	$(error "No wmake found, please activate an OpenFOAM environment.")
endif

# OpenFOAM Wrapper
# -------------------------------------------------------------------------------
# This target requires an activated OpenFOAM environment with "wmake" script avaliable.
# The output file is automatically located in $(FOAM_USER_LIBBIN) environment variable.
# Please copy and paste if you need this interface elsewhere.
$(OF_SO): $(OF_SRC)/TF_OF_Predictor.C
	wmake $(OF_SRC)


# OpenFOAM Test Program
# -------------------------------------------------------------------------------
# This target requires an activated OpenFOAM environment with "wmake" script avaliable
# The output file is automatically located in $(FOAM_USER_APPBIN) environment variable.  
# Please copy and paste if you need it elsewhere.
$(OF_EXE): $(OF_TEST)/OpenFOAM_test_AplusB.C
	wmake $(OF_TEST)





