poc: poc.cpp
	mpiCC -ltbb -Wall poc.cpp -o build/poc

run:
	mpirun -np 4 build/poc

.PHONY: run
