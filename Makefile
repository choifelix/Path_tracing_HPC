CC=mpicc -Wall -O3

CFLAGS=-Iinc

#LDFLAGS=-lm 

#BIN=pathtracer

all : $(BIN)
	mpicc -Wall -o pathtracer pathtracer_MPI.c -fopenmp -lm -O3

# % : %.c
# 	$(CC) -o $@ $^ $(LDFLAGS)

clean :
	rm -f $(BIN) *.o *~

exec:
	./pathtracer

static: $(BIN)
	mpicc -o pathtracer pathtracer_MPI_static.c -fopenmp -lm -O3

omp:
	mpicc -o pathtracer pathtracer_MPI_openMP.c -fopenmp -lm -O3






