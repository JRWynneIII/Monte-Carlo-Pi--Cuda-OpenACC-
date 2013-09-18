#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>

__global__ void kernel(int* count_d, float* randomnums)
{
	int i;
	double x,y,z,xs,ys,zs;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	i = tid;
	i = i + i;
	int xidx, yidx, sxidx, syidx = 0;
	int lower = (tid * 4);
	int upper = lower +3;
	xidx = lower;
	yidx = upper - 2;
	sxidx = upper - 1;
	syidx = upper;
	x = randomnums[xidx];
	y = randomnums[yidx];
	xs = randomnums[sxidx];
	ys = randomnums[syidx];
	z = ((x*x)+(y*y));
	zs = ((xs*xs)+(ys*ys));

	if (z<=1)
		count_d[i] = 1;
	else
		count_d[i] = 0;	
	if (zs<=1)
		count_d[i +1] = 1;
	else
		count_d[i+1] = 0;	
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
	int niter = 100000;
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

	int threads = 1000;
	int blocks = 50;
	int* count_d;
	int *count = (int*)malloc(niter*sizeof(int));
	unsigned int reducedcount = 0;
	cudaMalloc((void**)&count_d, (niter)*sizeof(int));
	CUDAErrorCheck();
	//one point per thread
	kernel <<<blocks, threads>>> (count_d, randomnums);
	cudaDeviceSynchronize();
	CUDAErrorCheck();
	cudaMemcpy(count, count_d, niter*sizeof(int), cudaMemcpyDeviceToHost);
	int i = 0;
	//reduce array into int
	for(i = 0; i<niter; i++)
	{
		reducedcount += count[i];
		printf("count[%d]:\t %d\n", i, count[i]);
	}
	cudaFree(randomnums);
	cudaFree(count_d);
	free(count);

	pi = ((double)reducedcount/niter)*4.0;
	printf("Pi: %f\n", pi);

	return 0;
}
