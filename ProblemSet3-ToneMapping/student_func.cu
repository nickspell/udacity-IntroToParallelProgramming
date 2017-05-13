/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "device_launch_parameters.h"
//#include "reference_calc.cpp"
#include <stdio.h>
#include <float.h>
#include <limits.h>

const int BLOCK_SIZE = 1024;

__device__ float _min(float a, float b) {
	return a < b ? a : b;
}

__device__ float _max(float a, float b) {
	return a > b ? a : b;
}

__global__ void minmax_reduce(float* d_out, const float * d_in, int input_size,bool isMin) {
	
	extern __shared__ float sdata[];
	
	int tid = threadIdx.x;
	int global_id = tid + blockDim.x*blockIdx.x;
	
	if (global_id >= input_size) { sdata[tid] = d_in[0]; } //dummy init (does not modify the final result)
	else sdata[tid] = d_in[global_id];
	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s>>=1){
		if (tid < s) sdata[tid] = isMin ? _min(sdata[tid], sdata[tid + s]) : _max(sdata[tid], sdata[tid + s]);
		__syncthreads();
	}
	if (tid == 0) {
		d_out[blockIdx.x] = sdata[0];
	}
}



__global__ void histo_atomic(unsigned int* out_histo,  const float * d_in, int numBins, int input_size, float minVal, float rangeVals) {
	int tid = threadIdx.x;
	int global_id = tid + blockDim.x*blockIdx.x;
	if (global_id >= input_size) return;
	int bin  = ((d_in[global_id] - minVal)*numBins) / rangeVals;
	bin = bin == numBins ? numBins - 1 : bin; //max value bin is the last of the histo
	atomicAdd(&(out_histo[bin]), 1);
}


//--------HILLIS-STEELE SCAN----------
//Optimal step efficiency (histogram is a relatively small vector)
//Works on maximum 1024 (Pascal) elems vector.
__global__ void scan_hillis_steele(unsigned int* d_out,const unsigned int* d_in, int size) {
	extern __shared__ unsigned int temp[];
	int tid = threadIdx.x;
	int pout = 0,pin=1;
	temp[tid] = tid>0? d_in[tid-1]:0; //exclusive scan
	__syncthreads();

	//double buffered
	for (int off = 1; off < size; off <<= 1) {
		pout = 1 - pout;
		pin = 1 - pout;
		if (tid >= off) temp[size*pout + tid] = temp[size*pin + tid]+temp[size*pin + tid - off];
		else temp[size*pout + tid] = temp[size*pin + tid];
		__syncthreads();
	}
	d_out[tid] = temp[pout*size + tid];
}


float reduce(const float* const d_logLuminance, int input_size,bool isMin) {
	int threads = BLOCK_SIZE;
	float* d_current_in = NULL;
	int size = input_size;
	int blocks = ceil(1.0f*size / threads); 
	while (true) {
		//allocate memory for intermediate results
		//printf("Size %d blocks %d\n", size,blocks);
		float* d_out;
		checkCudaErrors(cudaMalloc(&d_out, blocks * sizeof(float)));
		//call reduce kernel: if first iteration use original vector, otherwise use the last intermediate result.
		if (d_current_in == NULL) minmax_reduce << <blocks, threads, threads * sizeof(float) >> > (d_out, d_logLuminance, size, isMin);
		else minmax_reduce << <blocks, threads, threads * sizeof(float) >> > (d_out, d_current_in, size, isMin);;
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//free last intermediate result
		if (d_current_in != NULL) checkCudaErrors(cudaFree(d_current_in));

		if (blocks == 1) {
			//end of reduction reached
			float h_out;
			checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
			return h_out;
		}
		size = blocks;
		blocks = ceil(1.0f*size / threads); 
		if (blocks == 0)blocks++;
		d_current_in = d_out;//point to new intermediate result
		
	}
	
}


unsigned int* compute_histogram(const float* const d_logLuminance, int numBins, int input_size, float minVal, float rangeVals) {
	unsigned int* d_histo;
	checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));
	int threads = BLOCK_SIZE;
	int blocks = ceil(1.0f*input_size / threads);
	histo_atomic << <blocks, threads >> >(d_histo, d_logLuminance, numBins, input_size, minVal, rangeVals);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	return d_histo;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	//1. Reduce
	int input_size = numRows*numCols;
	min_logLum = reduce(d_logLuminance, input_size, true);
	max_logLum = reduce(d_logLuminance, input_size, false);
	//printf("%f %f\n", min_logLum, max_logLum);

	//2. Range
	float range = max_logLum - min_logLum;

	//3. Histogram
	unsigned int* d_histo=compute_histogram(d_logLuminance, numBins, input_size, min_logLum, range);

	//4. CDF (scan)
	//Assumption: numBins<=1024
	scan_hillis_steele << <1, numBins, 2*numBins*sizeof(unsigned int) >> > (d_cdf,d_histo, numBins);

	checkCudaErrors(cudaFree(d_histo));

}
