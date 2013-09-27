all:
	nvcc -G -g -arch=sm_30 -lcurand -c oaccreduce.cu -o cuda.o
	CC -g -lcurand -acc accreduce.cpp cuda.o -o /tmp/work/wyn/a.out
