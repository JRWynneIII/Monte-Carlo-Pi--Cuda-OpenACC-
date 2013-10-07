#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>

__global__ void kernel(int* count_d, float* randomnums)
{
	int i;
	double x,y,z;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	i = tid;
	int xidx = 0, yidx = 0;

	xidx = (i+i);
	yidx = (xidx+1);

	x = randomnums[xidx];
	y = randomnums[yidx];
	z = ((x*x)+(y*y));

	if (z<=1)
		count_d[tid] = 1;
	else
		count_d[tid] = 0;	
}

void CUDAErrorCheck()
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{	
		printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
		exit(0);
	}
}

extern "C" int* launch(int threads, int blocks)
{
	int niter = 230400;
	float *randomnums;
	cudaMalloc((void**)&randomnums, (2*niter)*sizeof(float));
	// Use Rand to generate an array of random numbers on the device
	int status;
	curandGenerator_t gen;
	status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
	status |= curandSetPseudoRandomGeneratorSeed(gen, 4294967296ULL^time(NULL));
	status |= curandGenerateUniform(gen, randomnums, (2*niter));
	status |= curandDestroyGenerator(gen);
	if (status != CURAND_STATUS_SUCCESS)
	{
		printf("CuRand Failure\n");
		exit(EXIT_FAILURE);
	}

	int* count_d;
	int *count = (int*)malloc(blocks*threads*sizeof(int));

	cudaMalloc((void**)&count_d, (blocks*threads)*sizeof(int));
	CUDAErrorCheck();

	kernel <<<blocks, threads>>> (count_d, randomnums);
	cudaDeviceSynchronize();
	//reduction <<<blocks, threads, threads*sizeof(int)>>> (count_d);
	cudaDeviceSynchronize();
	CUDAErrorCheck();


	cudaFree(randomnums);
	free(count);

	return count_d;
}
