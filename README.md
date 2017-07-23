# udacity-IntroToParallelProgramming
CS344 - Introduction To Parallel Programming course (Udacity) proposed solutions

Testing Environment: Visual Studio 2015 x64 + nVidia CUDA 8.0 + OpenCV 3.2.0

For each problem set, the core of the algorithm to be implemented is located in the _students_func.cu_ file.

## Problem Set 1 - RGB2Gray: 
### Objective
Convert an input RGBA image into grayscale version (ignoring the A channel).
### Topics
Example of a **map** primitive operation on a data structure.

## Problem Set 2 - Blur
### Objective
Apply a Gaussian blur convolution filter to an input RGBA image (blur each channel independently, ignoring the A channel).
### Topics
Example of a **stencil** primitive operation on a 2D array. Use of the **shared memory** in order to speed-up the algorithm. Both global memory and shared memory based kernels are provided, the latter providing approx. 1.6 speedup over the first.

## Problem Set 3 -Tone Mapping
### Objective
Map a High Dynamic Range image into an image for a device supporting a smaller range of intensity values.
### Topics
- Compute range of intensity values of the input image: min and max **reduce** implemented.
- Compute **histogram** of intensity values (1024-values array)
- Compute the cumulative ditribution function of the histogram: Hillis & Steele **scan** algorithm (step-efficient, well suited for small arrays like the histogram one). 

## Problem Set 4 - Red eyes removal
### Objective
Remove red eys effect from an inout RGBA image (it uses Normalized Cross Correlation against a training template).
### Topics
Sorting algorithms with GPU: given an input array of NCC scores, sort it in ascending order: **radix sort**. For each bit:
- Compute a predicate vector (0:false, 1:true)
- Performs **Bielloch Scan** on the predicate vector (for both false and positive cases)
- From Bielloch Scan extracts: an histogram of predicate values [0 numberOfFalses], an offset vector (the actual result of scan)
- A move kernel computes the new index of each element (using the two structures above), and moves it.

## Problem Set 5 - Optimized histogram computation
### Objective
Improve the histogram computation performance on GPU over the simple global atomic solution.
### Topics
**Per-block** histogram computation. Each block computes his own histogram in shared memory, and histograms are combined at the end in global memory (more than 7x speedup over global atomic implementation, while being relatively simple). 

## Problem Set 6 - Seamless Image Cloning
### Objective
Given a target image (e.g. a swimming pool), do a seamless attachment of a source image mask (e.g. an hyppo).
### Topics
The algorithm consists into performing Jacobi iterations on the source and target image to blend one with the other.
- Given the mask, detect the interior points and the boundary points
- Since the algorithm has to be performed only on the interior points, compute the **bounding box** of the mask region to restrict the Jacobi iterations on a subimage.
- Split the images in the R,G and B channels.
- Run 800 Jacobi iterations on each channel. The code makes use of **CUDA Streams** to run concurrently the same kernel on the 3 different channels (speedup of 3x on my machine, of 1.5x on the Udacity machine). The Jacobi kernel makes extensive use of shared memory, so the number of threads per block has been reduced to maximize SM's occupancy.
- Recombine the 3 channels to form the output image.
