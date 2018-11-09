the_offloadinator: the_offloadinator.cpp
	mkdir -p build
	mpiCC -ltbb -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -g the_offloadinator.cpp -o build/the_offloadinator

run:
	mpirun -n 3 build/the_offloadinator

dbg:
	mpirun -n 5 -oversubscribe xterm -e gdb build/the_offloadinator

.PHONY: run dbg
