#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

extern "C" int* launch(int threads, int blocks);

int reduce(int blocks, int* count_d )
{
	int sum = 0;
	int i = 0;
	#pragma acc parallel present(count_d) reduction(+:sum)
	{
		for(i = 0; i<blocks; i++)
		{
			sum += count_d[i];
		}
	}
	return sum;
}

int main()
{
	int niter = 230400;
	double pi;
	int threads = 512;
	int blocks = 450;
	unsigned int reducedcount = 0;
	int *count_d;
	count_d = launch(threads, blocks);
	int sum = reduce(blocks, count_d); 
	pi = ((double)reducedcount/niter)*4.0;
	printf("Pi: %f\n", pi);
}
