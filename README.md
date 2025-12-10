Test for writing simple Paraview-readable files using C++. 

Compile with:
mpicxx -std=c++11 -O3 -o test_paraview test.cpp
Run with:
mpirun -np 4 ./test_paraview                    
