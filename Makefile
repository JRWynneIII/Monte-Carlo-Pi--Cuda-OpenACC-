all:
	nvcc -G -g -arch=sm_30 -lcurand -c oaccreduce.cu -o cuda.o
	CC -g -Minfo=all -ta=nvidia:5.0,cc3x -lcurand -acc accreduce.cpp cuda.o -o /tmp/work/wyn/a.out
