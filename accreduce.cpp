#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

extern "C" int* launch(int threads, int blocks);

int reduce(int niter, int *count_d )
{
	int sum = 0;
	int i = 0;
       	#pragma acc parallel loop reduction(+:sum) private(i) deviceptr(count_d) 
		for(i = 0; i<niter; i++)
		{
			sum += count_d[i];
		}

	printf("sum[%d]:\t%d\n", i, sum);
	return sum;
}

int main()
{
	int niter = 230400;
	double pi;
	int threads = 512;
	int blocks = 450;
	int *count_d;
	count_d = launch(threads, blocks);
	int sum = reduce(niter, count_d); 
	std::cout<<sum<<std::endl;
	pi = ((double)sum/niter)*4.0;
	printf("Pi: %f\n", pi);
}
