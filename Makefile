poc: poc.cpp
	mpiCC -ltbb -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic poc.cpp -o build/poc

run:
	mpirun -np 4 build/poc

.PHONY: run
