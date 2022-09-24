# This makefile is used to compile CFL3D_ML code.
# written by Weishuo Liu.


# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean clean_seq_ml clean_mpi_ml cfl3d_seq_ml cfl3d_mpi_ml

all: cfl3d_seq_ml

cfl3d_seq_ml:
	make -f Makefile.seq

cfl3d_mpi_ml:
	make -f Makefile.mpi


clean_seq_ml:
	make -f Makefile.seq clean

clean_mpi_ml:
	make -f Makefile.mpi clean

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

clean:
	make -f Makefile.seq clean
	make -f Makefile.mpi clean


