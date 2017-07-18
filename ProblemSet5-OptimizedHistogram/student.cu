/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>

const int N_THREADS =  1024;



__global__
void naiveHisto(const unsigned int* const vals, //INPUT
	unsigned int* const histo,      //OUPUT
	int numVals)
{
	int tid = threadIdx.x;
	int global_id = tid + blockDim.x*blockIdx.x;
	if (global_id >= numVals) return;
	atomicAdd(&(histo[vals[global_id]]), 1);
}

__global__
void perBlockHisto(const unsigned int* const vals, //INPUT
	unsigned int* const histo,      //OUPUT
	int numVals,int numBins) {

	extern __shared__ unsigned int sharedHisto[]; //size as original histo

	//coalesced initialization: multiple blocks could manage the same shared histo
	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		sharedHisto[i] = 0;
	}

	__syncthreads();

	int globalid = threadIdx.x + blockIdx.x*blockDim.x;
	atomicAdd(&sharedHisto[vals[globalid]], 1);
	
	__syncthreads();

	for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
		atomicAdd(&histo[i], sharedHisto[i]);
	}


}



void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

	int blocks = ceil(numElems / N_THREADS);

	//naiveHisto <<< blocks, N_THREADS >>> (d_vals, d_histo, numElems);


	//more than 7x speedup over naiveHisto
	perBlockHisto << <blocks, N_THREADS, sizeof(unsigned int)*numBins >> > (d_vals, d_histo, numElems, numBins);

  //if you want to use/launch more than one kernel,
  //feel free

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
