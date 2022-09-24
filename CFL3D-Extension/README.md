# CFL3D-Extension
Application of the **NN_Pred** in **CFL3D**, which is a compressible RANS solver based on structrued grid. The original CFL3D solver can be found [here](https://github.com/nasa/CFL3D):

The folder consists of an original cfl3d solver (with modified output field) and the ML version which substitute the original eddy-viscosity with the ML prediction. The output fields in the `plot3dq.q` follows the sequence:
```
{'rho': 1, 'u': 2, 'v': 3, 'w': 4, 'p': 5, 'omega': 6, 'k': 7, 'vist': 8}
```

## Compilation of the CFL3D solvers
With the compiled Fortran API of the core predictor, one can compile the solver via:
```sh
cd CFL3D-Extension  # Go to the OF Extension dir

# build CFL3D with the ML Predictor extension 
make cfl3d_seq_ori    # Original cfl3d serial version
make cfl3d_mpi_ori    # Original cfl3d parallel version
make cfl3d_seq_ml     # ML-RANS cfl3d serial version
make cfl3d_mpi_ml     # ML-RANS cfl3d parallel version
```

## Run the turbulent channel case:
The corresponding solvers needs to be copied into the case folder to run the simulation, which can be done by the makefile scrpt:
```sh
make install
```
Then the simulations of turbulent channel flow can be conducted by:
```sh
cd Channel_Cases/kwSST_Re180 # Go to the original kwSST case
./cfl3d_seq_ori < cfl3d.inp &   # Run the simulation
```
```sh
cd Channel_Cases/ML_Re180 # Go to the ML-RANS case
./cfl3d_seq_ml < cfl3d.inp &   # Run the simulation
```
```sh
cd Channel_Cases/kwSST_Re180_2blk # Go to the original kwSST parallel case
mpiexec -np 3 ./cfl3d_mpi_ori < cfl3d.inp &   # Run the parallel simulation, need one more core to be host
```
```sh
cd Channel_Cases/ML_Re180_2blk # Go to the ML-RANS parallel case
mpiexec -np 3 ./cfl3d_mpi_ml < cfl3d.inp &    # Run the parallel simulation
```

