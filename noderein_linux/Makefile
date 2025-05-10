CXX=g++
FLAGS=-Wall -O3 -march=native -g -Wno-deprecated -I. 
LIBS=-lboost_program_options
BIN=reverse-greedy

# uncomment the following line if you have an openmp-compliant compiler

OMP=0

ifeq ($(OMP),1)
	FLAGS+=-fopenmp -DOMP
	LIBS+=-lgomp
else
	FLAGS+=-Wno-unknown-pragmas

endif

all: $(BIN)

reverse-greedy: reverse-greedy.cpp real_type.hpp
	${CXX} ${FLAGS} reverse-greedy.cpp ${LIBS} -o $@
clean:
	rm -f $(BIN) *.o
