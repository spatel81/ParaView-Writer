Test for writing simple Paraview-readable files using C++. 

```sh
module load PrgEnv-gnu
```

* Compile with:
CC -std=c++11 -O3 -o test.exe main.cpp
* Run with:
mpiexec -np 1 ./test.exe                   
