# Path_tracing_HPC

mpicc -o pathtracerMPI pathtracer_MPI.c -lm

mpirun -n 3 -hostfile hostfile --map-by node ./pathtracerMPI

 display /tmp/3520621/image.ppm