# This makefile is used to compile CFL3D_ML code.
# written by Weishuo Liu.


# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean clean_seq_ori clean_mpi_ori cfl3d_seq_ori cfl3d_mpi_ori

all: cfl3d_seq_ori

cfl3d_seq_ori:
	make -f Makefile.seq

cfl3d_mpi_ori:
	make -f Makefile.mpi

clean_seq_ori:
	make -f Makefile.seq clean

clean_mpi_ori:
	make -f Makefile.mpi clean

clean:
	make -f Makefile.seq clean
	make -f Makefile.mpi clean


