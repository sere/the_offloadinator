poc: poc.cpp
	mpiCC -ltbb -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic poc.cpp -o build/poc

run:
	mpirun -np 3 build/poc

dbg:
	mpirun -np 3 xterm -e gdb build/poc

.PHONY: run dbg
