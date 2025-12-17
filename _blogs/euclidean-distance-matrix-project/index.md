---
layout: post
title: Kernel Optimization of Euclidean Distance Matrix for 2D coordinate.
description: CUDA Kernel development and optimization for euclidean distance matrix calculations.
skills: 
  - CUDA Programming.
  - C++/C Programming.
  - CUDA Kernel Profiling.
  - Memory Hierarchy. 
  - CUDA Pipleines.

---

---
## Introduction

Euclidean matrix distance is an nxn matrix representing the euclidean distance between set of n points in Euclidean space. Euclidean distance matrix has so many application in machine learning, Engineering and robotics and image processing. This technical blog will discuss step by step on how to optimize CUDA kernel for Euclidean Distance Matrix for 2D coordinate points. 

## Memory bound vs Compute bound for Euclidean distance Matrix

The eucildean distance calculation between 2D cooordinate points involves two substraction, two multiplication, one addition and one sqrt. According to this [post](https://forums.developer.nvidia.com/t/performance-tweak-for-single-precision-square-root/173274), sqrt() function involves about four to five floating point operation. Therefore, one euclidean distance calculation between 2D coordinates points involves 9 floating point operation. For 30336 2D coordinate points, which is what was used in this optimization. 

   1. Total FLOPS: 30336<sup>2</sup> * 9 FLOPS = 8282456064B(8.28GFLOPS).
   2. Minimum total data to read: 30336 * 8B = 242688B(242KB).
   3. Total data to write: 30336<sup>2</sup> * 4B = 3681091584B(3.68GB).
   
The Nvidia RTX 5070 Ti has a memory bandwith of 896GB/sec and has a fp32 compute throughput of 41TFLOPS. Therefore, the theoretical time for the calculation is 0.2 milliseconds while the theoretical total time for data read and write is 4.1 milliseconds assuming the both total read and write is 3.68GB. This simple theoretical calculation shows that the euclidean distance matrix calculation is memory-bound.       

## Kernel 1: Naive implementation

The naive implementation involves each GPU thread computing the euclidean distance between one coordinate and every other coordinate, which means each GPU thread will compute n euclidean distance out of n x n euclidean distance matrix. Therefore, each block of thread compute (n * blocksize) euclidean distance. In CUDA programming model, threads are execute by warp, which is a group of 32 threads that executes the same instruction simultaneously(Single Instruction Multiple Threads). The minimum memory transaction by a warp is 32 bytes adjacent data (also known sector), which means a warp need four memory transaction to write 32 float data type to global memory if the data are adjacent. For kernel 1, where the data written to global memory are not adjacent(coalesced), a warp will need 32 memory transaction(1024bytes) to write 32 float data type to global memory, which leads to the waste of 992 bytes of memory bandwith per warp. The kernel 1 function has a compute throughput and memory throughput of 475 GFLOPS and 233 GB/s based on nsight compute. The actual memory throughput is lower when the wasted memory bandwith is considered for global memory write, in which the actual memory throughput is 173 GB/s ((3.68/4.97) * 233 GB/s).    

 
```cuda
// Naive kernel function

__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA) {
    
   size_t gid =  blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < NUMDATA) {
    	size_t index = gid * NUMDATA;
    	for (int i = 0; i < NUMDATA; i++)  {
           float x_co =  (cordinates[gid].x - cordinates[i].x);
           float y_co =  (cordinates[gid].y - cordinates[i].y);
           float pow_xco = x_co * x_co;
           float pow_yco = y_co * y_co;
       	   float pow_plus = sqrt(pow_yco+pow_xco);
           euclideanDistance[index+i] = pow_plus;
         }
    }
    
}
```


## Kernel 2: Global Memory Coalescing

The kernel 2 function involves each warp writing the results of the euclidean distance calculations to 32 adjacent float data memory (4 adjacent sectors), which means all warps in a block writes the results of the calculations to 256 adjacent float data memory (32 adjacent sector for a blocksize of 256).
    
```cuda
// Kernel function with coalesced global memory write
__global__  void euclideanMatrix(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA) {
    
   size_t gid_start =  blockIdx.x * blockDim.x; 
   size_t blocksize =   blockDim.x*blockDim.y*blockDim.z;
   size_t index;
   size_t real_gid;
   size_t k;
   size_t j;
   for (int i = threadIdx.x; i < NUMDATA*blocksize; i+=blocksize)  {
       j = i / NUMDATA;
       real_gid =  j + gid_start;
       if (real_gid >= NUMDATA) {
           continue;
       }
       k = i - (j * NUMDATA); 
       index = real_gid * NUMDATA;	   
       float x_co =  (cordinates[real_gid].x - cordinates[k].x);
       float y_co =  (cordinates[real_gid].y - cordinates[k].y);
       float pow_xco = x_co * x_co;
       float pow_yco = y_co * y_co;
       float pow_plus = sqrt(pow_yco+pow_xco);
       euclideanDistance[index+k] = pow_plus;
  }
    
}
```
 

## Kernel 3: Shared Memory Cache-Blocking

```cuda
// Kernel function with shared memory
__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread) {
    
   size_t gid_start =  blockIdx.x *  blockDim.x;
   size_t gid =  blockIdx.x * blockDim.x + threadIdx.x;
   extern  __shared__ LocationPrim locations [];
   int blocksize = blockDim.x * blockDim.y * blockDim.z;          
   size_t numofDataperBatch = (numDataPerThread) * blocksize;
   auto numBatchToFetch = [&](int batchfetched) -> int {	   
     return ((NUMDATA - batchfetched) >= numofDataperBatch) ? numofDataperBatch : (NUMDATA - batchfetched);
   };
      
   size_t index;
   size_t real_gid;
   size_t t = 0;
   size_t k;
   size_t dataSub;
   size_t ref_index;
   size_t d; 
   size_t dataFetchSize;  	  
   size_t threadId = threadIdx.x;
   size_t totalDataCompute; 
   if (gid < NUMDATA) {
       locations[numofDataperBatch + threadId] = cordinates[gid];    
 
   } 
    
   for (int i = 0; i < NUMDATA; i+=numBatchToFetch(i)) {

       dataFetchSize = numBatchToFetch(i);  	  
       for (size_t n = threadId, m = i + threadId; n < dataFetchSize; n+=blocksize, m+= blocksize) {
           //locations[n] = cordinates[m];
	   __pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
       } 
       __pipeline_commit();
       __pipeline_wait_prior(0);
       __syncthreads();
       
       t = 0;
       totalDataCompute = dataFetchSize*blocksize;       
       //count = threadIdx.x;
       for (size_t z = threadId, c = i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
          t  = z/dataFetchSize;   
          real_gid =  t + gid_start;
          if (real_gid >= NUMDATA) {
            continue;
          }
	      dataSub = t * dataFetchSize;
          k = c - dataSub; 
          index = real_gid*NUMDATA;
          d = z - dataSub;
	      ref_index = numofDataperBatch + t;  
          float x_co =  (locations[ref_index].x - locations[d].x);
          float y_co =  (locations[ref_index].y - locations[d].y);
	      float pow_xco = x_co * x_co;
          float pow_yco = y_co * y_co;
          float pow_plus = sqrt(pow_yco+pow_xco);
          euclideanDistance[index+k] = pow_plus;
       }  
      __syncthreads();
	 
      } 
}
```


## Kernel 4: Instruction Optimization

```cuda
// Kernel function with instruction optimization

__global__  void euclideanMatrixDynamicSharedMemory(LocationPrim *cordinates, float* euclideanDistance, size_t NUMDATA, int numDataPerThread) {
   size_t gid_start =  blockIdx.x * blockDim.x;
   size_t gid =  blockIdx.x * blockDim.x + threadIdx.x;
   extern  __shared__ LocationPrim locations [];
   int blocksize = blockDim.x * blockDim.y * blockDim.z;      
   size_t numofDataperBatch = (numDataPerThread) * blocksize;
   size_t numRef = ((numDataPerThread+1) * blocksize);
   auto numBatchToFetch = [&](int batchfetched) -> int {	   
     return ((NUMDATA - batchfetched) >= (numofDataperBatch + blocksize)) ? numofDataperBatch : (NUMDATA - batchfetched);
   };
   size_t index;
   size_t real_gid;
   size_t t = 0;
   size_t k;
   size_t dataSub;
   size_t ref_index;
   size_t d; 
   size_t dataFetchSize;  	  
   size_t threadId = threadIdx.x;
   size_t totalDataCompute;
   if (gid < NUMDATA) {
       locations[numRef + threadId] = cordinates[gid];    
   } 
   for (int i = 0; i < NUMDATA; i+=numBatchToFetch(i)) {
       dataFetchSize = numBatchToFetch(i);  	  
       for (size_t n = threadId, m = i + threadId; n < dataFetchSize; n+=blocksize, m+= blocksize) {
           //locations[n] = cordinates[m];
	   __pipeline_memcpy_async(&locations[n], &cordinates[m], sizeof(LocationPrim));
       } 
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
       t = 0;
       totalDataCompute = dataFetchSize*blocksize;
       for (size_t z = threadId, c = i + threadId; z < totalDataCompute; z+=blocksize, c+=blocksize)  {
          if (z >= ((t + 1) * dataFetchSize)) {
               t = t + 1;
          } 
          real_gid =  t + gid_start; 
          if (real_gid >= NUMDATA) {
            continue;
          }
	  dataSub = t * dataFetchSize;
          k = c - dataSub; 
          index = real_gid*NUMDATA;
          d = z - dataSub;
	  ref_index = numRef + t;  
          float x_co =  (locations[ref_index].x - locations[d].x);
          float y_co =  (locations[ref_index].y - locations[d].y);
	  float pow_xco = x_co * x_co;
          float pow_yco = y_co * y_co;
          float pow_plus = sqrt(pow_yco+pow_xco);
          euclideanDistance[index+k] = pow_plus;
       }  
      __syncthreads();	 
   }
}
```

 

## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [SOL Analysis with NVIDIA Nsight Compute](https://www.youtube.com/watch?v=uHN5fpfu8As)
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
