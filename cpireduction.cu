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

__global__ void reduction(int* count_d)
{
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	sdata[tid] = count_d[i];
	__syncthreads();
	//reduce all in sdata[] to one int at sdata[0]
	for (int a = blockDim.x/2; a>0 ;a>>=1)
	{
		if (tid<a)
		{
			sdata[tid] += sdata[tid+a];
		}
	__syncthreads();
	}
	if (tid == 0)
	{
		count_d[blockIdx.x] = sdata[0];
	}
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

int main(int argc,char* argv[])
{
	int niter = 100489;
	float *randomnums;
	double pi;
	cudaMalloc((void**)&randomnums, (2*niter)*sizeof(float));
	// Use CuRand to generate an array of random numbers on the device
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

	int threads = 317;
	int blocks = 317;
	int* count_d;
	int *count = (int*)malloc(blocks*threads*sizeof(int));
	unsigned int reducedcount = 0;

	cudaMalloc((void**)&count_d, (blocks*threads)*sizeof(int));
	CUDAErrorCheck();

	kernel <<<blocks, threads>>> (count_d, randomnums);
	cudaDeviceSynchronize();
	reduction <<<blocks, threads, threads*sizeof(int)>>> (count_d);
	cudaDeviceSynchronize();
	CUDAErrorCheck();

	cudaMemcpy(count, count_d, blocks*sizeof(int), cudaMemcpyDeviceToHost);
	int i = 0;
	//reduce array into int
	for(i = 0; i<=blocks; i++)
	{
		reducedcount += count[i];
		printf("count[%d]:\t%d\n", i, count[i]);
	}

	cudaFree(randomnums);
	cudaFree(count_d);
	free(count);

	pi = ((double)reducedcount/niter)*4.0;
	printf("Pi: %f\n", pi);

	return 0;
}
