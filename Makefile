poc: poc.cpp
	mkdir -p build
	mpiCC -ltbb -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -g poc.cpp -o build/poc

run:
	mpirun -n 3 build/poc

dbg:
	mpirun -n 5 -oversubscribe xterm -e gdb build/poc

.PHONY: run dbg
