# This makefile is used to compile CFL3D_ML code.
# written by Weishuo Liu.

# Targets:
# -------------------------------------------------------------------------------
.PHONY: all clean clean_seq_ml clean_mpi_ml clean_seq_ori clean_mpi_ori cfl3d_seq_ori cfl3d_mpi_ori cfl3d_seq_ml cfl3d_mpi_ml install

all: cfl3d_seq_ori cfl3d_mpi_ori cfl3d_seq_ml cfl3d_mpi_ml

cfl3d_seq_ori:
	cd Solvers/CFL3D_ORI/ && make cfl3d_seq_ori

cfl3d_mpi_ori:
	cd Solvers/CFL3D_ORI/ && make cfl3d_mpi_ori

cfl3d_seq_ml:
	cd Solvers/CFL3D_ML/ && make cfl3d_seq_ml

cfl3d_mpi_ml:
	cd Solvers/CFL3D_ML/ && make cfl3d_mpi_ml

clean_seq_ori:
	cd Solvers/CFL3D_ORI/ && make clean_seq_ori

clean_mpi_ori:
	cd Solvers/CFL3D_ORI/ && make clean_mpi_ori

clean_seq_ml:
	cd Solvers/CFL3D_ML/  && make clean_seq_ml

clean_mpi_ml:
	cd Solvers/CFL3D_ML/  && make clean_mpi_ml

clean:
	cd Solvers/CFL3D_ORI/ && make clean
	cd Solvers/CFL3D_ML/  && make clean


install: all
	cp Solvers/CFL3D_ORI/outputs/cfl3d_seq_ori/bin/cfl3d_seq_ori Channel_Cases/kwSST_Re180/
	cp Solvers/CFL3D_ORI/outputs/cfl3d_mpi_ori/bin/cfl3d_mpi_ori Channel_Cases/kwSST_Re180_2blk/
	cp Solvers/CFL3D_ML/outputs/cfl3d_seq_ml/bin/cfl3d_seq_ml Channel_Cases/ML_Re180/
	cp Solvers/CFL3D_ML/outputs/cfl3d_mpi_ml/bin/cfl3d_mpi_ml Channel_Cases/ML_Re180_2blk/


