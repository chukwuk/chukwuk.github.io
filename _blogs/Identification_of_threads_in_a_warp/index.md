---
layout: post
title: How CUDA threads are grouped in a CUDA Warp.
description: GPU microbenckmarking was used to identify group of threads in a Warp.
skills: 
  - CUDA Programming.
  - C++/C Programming.
  - Shared and Global memory.
  - CUDA Warps 

---

---
## Introduction  

A CUDA warp is a group of 32 threads that executes the same instruction. Understanding how threads are grouped in a warp is important for warp tiling used in optimizing matrix mutiplication, which is a common computation in deep learning. In this technical blog, I will discuss how microbenchmarking is used identify 32 threads that are grouped as a warp. According to this [blog]( https://siboehm.com/articles/22/CUDA-MMM), threads are grouped as a warp based on consecutive threadId. 
```cuda
threadId = threadId.x+ (blockDim.x * threadId.y) + (blockDim.x * blockDim.y * threadId.z)
``` 
In this technical blog, I will be microbenchmarking NVIDIA RTX 5070 Ti to identify which group of threads belong to a warp. This [paper](https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf) has more information about GPU microbenchmarking.

## Microbenchmarking 

CUDA programming model and NVIDIA hardware architecture never allows two different warps to access the shared memory simultaneously in a single clock cycle. Consequently, the number of clock cycles was measured for loading of data from the global memory(gmem) to shared memory(smem), since when one warp is accessing the shared memory, the other warp will be stalled and this will lead to different number of clock cycles for loading the data from the gmem to smem. The NVIDIA RTX 5070 ti has 128 cores per SM, so 128 blocksize was used to ensure the four warp can be executed in a single clock cycle, which will lead to stalling of the other three warps when loading data from gmem to smem. Threads in the same warp will have the same number of clock cycles for loading the data from gmem to smem, which is used to identify threads in the same warp. The struct below was used to collect information about the number of clock cycles, threadId.x, threadId.y and threadId.z for each of the thread. 

```cuda
struct threadProperties {
    unsigned long long time;
    int thread_x;
    int thread_y;
    int thread_z;
    int value;
};
```

## Warp in 1D block

For 1D block, Warp one has threadId.x ∈ {0,1,2...31, Warp two has threadId.x ∈ {32,33,34.....63}, Warp three has threadId.x ∈ {64,65,66....95}, Warp four has threadId.x ∈{95,96,97....127}. The microbenchmarking confirms the above grouping of threads in a warp, in which it shows that threads in the same warp have the same number of clock cycle for loading data from gmem to smem while threads in different warp have different number of clock cycles. The code is available on [Github repo](https://github.com/chukwuk/Identification_of_threads_in_a_Warp).   

```cuda
// A simple CUDA kernel.
__global__  void threadsInWarp(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  blockDim.x +  threadIdx.x;
   
   float copyvalue; 
   unsigned long long startTime = clock();  
   readtimer[threadIdx.x] = globalData[threadIdx.x];
    
   unsigned long long finishTime = clock();  

   copyvalue = readtimer[threadIdx.x];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   
}
```

```cuda
(threadId.x: 0) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 1) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 2) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 3) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 4) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 5) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 6) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 7) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 8) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 9) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 10) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 11) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 12) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 13) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 14) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 15) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 16) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 17) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 18) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 19) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 20) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 21) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 22) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 23) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 24) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 25) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 26) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 27) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 28) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 29) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 30) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 31) execution time for copying data from GMEM to SMEM: 2883 clock cycle
(threadId.x: 32) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 33) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 34) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 35) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 36) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 37) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 38) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 39) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 40) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 41) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 42) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 43) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 44) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 45) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 46) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 47) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 48) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 49) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 50) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 51) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 52) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 53) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 54) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 55) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 56) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 57) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 58) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 59) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 60) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 61) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 62) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 63) execution time for copying data from GMEM to SMEM: 2884 clock cycle
(threadId.x: 64) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 65) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 66) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 67) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 68) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 69) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 70) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 71) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 72) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 73) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 74) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 75) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 76) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 77) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 78) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 79) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 80) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 81) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 82) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 83) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 84) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 85) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 86) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 87) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 88) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 89) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 90) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 91) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 92) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 93) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 94) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 95) execution time for copying data from GMEM to SMEM: 2525 clock cycle
(threadId.x: 96) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 97) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 98) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 99) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 100) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 101) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 102) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 103) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 104) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 105) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 106) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 107) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 108) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 109) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 110) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 111) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 112) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 113) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 114) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 115) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 116) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 117) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 118) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 119) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 120) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 121) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 122) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 123) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 124) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 125) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 126) execution time for copying data from GMEM to SMEM: 2527 clock cycle
(threadId.x: 127) execution time for copying data from GMEM to SMEM: 2527 clock cycle
```

## Warp in 2D block

In the 2D block, the blockDim.x is 16, blockDim.y is 8 and blockDim.z is 1. Warp one has group of threads = {(threadId.x,threadId.y) in ℤ<sup>2</sup>, 0 &le; threadId.x &le; 16,0 &le; threadId.y &le; 1}, Warp two has group of threads = {(threadId.x,threadId.y)\in ℤ<sup>2</sup>, 0 &le; threadId.x &le; 16,2 &le; threadId.y &le; 3}, Warp three has group of threads = {(threadId.x,threadId.y) in ℤ<sup>2</sup>, 0 &le; threadId.x &le; 16,4 &le; threadId.y &le; 5}, Warp four has group of threads = {(threadId.x,threadId.y) in ℤ<sup>2</sup>, 0 &le; threadId.x &le; 16,6 &le; threadId.y &le; 7}. The microbenchmarking confirms the above grouping of threads in a warp, in which it shows that threads in the same warp have the same number of clock cycle for loading data from gmem to smem while threads in different warp have different number of clock cycle. 

```cuda
__global__  void threadsInWarp2D(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
 


   float copyvalue; 
   unsigned long long startTime = clock();  
   readtimer[gid] = globalData[gid];
    
   unsigned long long finishTime = clock();  

   copyvalue = readtimer[gid];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   threadsDev[gid].thread_y = threadIdx.y;   
}
```

```cuda
(threadId.x: 0, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 1, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 2, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 3, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 4, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 5, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 6, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 7, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 8, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 9, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 10, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 11, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 12, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 13, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 14, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 15, threadId.y: 0) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 0, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 1, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 2, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 3, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 4, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 5, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 6, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 7, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 8, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 9, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 10, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 11, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 12, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 13, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 14, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 15, threadId.y: 1) execution time for copying data from GMEM to SMEM: 3012 clock cycle
(threadId.x: 0, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 1, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 2, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 3, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 4, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 5, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 6, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 7, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 8, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 9, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 10, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 11, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 12, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 13, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 14, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 15, threadId.y: 2) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 0, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 1, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 2, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 3, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 4, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 5, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 6, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 7, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 8, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 9, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 10, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 11, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 12, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 13, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 14, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 15, threadId.y: 3) execution time for copying data from GMEM to SMEM: 3013 clock cycle
(threadId.x: 0, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 1, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 2, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 3, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 4, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 5, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 6, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 7, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 8, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 9, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 10, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 11, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 12, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 13, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 14, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 15, threadId.y: 4) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 0, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 1, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 2, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 3, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 4, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 5, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 6, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 7, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 8, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 9, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 10, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 11, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 12, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 13, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 14, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 15, threadId.y: 5) execution time for copying data from GMEM to SMEM: 2856 clock cycle
(threadId.x: 0, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 1, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 2, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 3, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 4, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 5, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 6, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 7, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 8, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 9, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 10, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 11, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 12, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 13, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 14, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 15, threadId.y: 6) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 0, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 1, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 2, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 3, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 4, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 5, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 6, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 7, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 8, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 9, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 10, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 11, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 12, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 13, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 14, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
(threadId.x: 15, threadId.y: 7) execution time for copying data from GMEM to SMEM: 2860 clock cycle
```

## Warp in 3D block

In the 3D block, the blockDim.x is 8, blockDim.y is 8 and blockDim.z is 2.

```cuda
__global__  void threadsInWarp3D(threadProperties* threadsDev, int* globalData) {
  
   	
   __shared__ int readtimer [128];
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   //size_t gid = blockIdx.x *  (blockDim.x *  blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
 
   //size_t gid =  blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * blockDim.y * blockDim.x) + 
    (threadIdx.y * blockDim.x) + threadIdx.x;
   size_t gid =  blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * blockDim.y * blockDim.x) + 
    (threadIdx.x * blockDim.y) + threadIdx.y;


   float copyvalue; 
   unsigned long long startTime = clock();  
   readtimer[gid] = globalData[gid];
    
   unsigned long long finishTime = clock();  

   copyvalue = readtimer[gid];

   // Calculate elapsed time
   
      
   unsigned long long GpuTime = finishTime - startTime;
   copyvalue++; 

   threadsDev[gid].value = copyvalue;
   threadsDev[gid].time = GpuTime;
   threadsDev[gid].thread_x = threadIdx.x;   
   threadsDev[gid].thread_y = threadIdx.y;   
   threadsDev[gid].thread_z = threadIdx.z;   
}
```

```cuda
(threadId.x: 0, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 1, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 2, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 3, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 4, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 5, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 6, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 7, threadId.y: 0, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 0, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 1, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 2, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 3, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 4, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 5, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 6, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 7, threadId.y: 1, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 0, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 1, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 2, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 3, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 4, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 5, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 6, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 7, threadId.y: 2, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 0, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 1, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 2, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 3, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 4, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 5, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 6, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 7, threadId.y: 3, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2870 clock cycle
(threadId.x: 0, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 1, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 2, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 3, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 4, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 5, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 6, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 7, threadId.y: 4, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 0, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 1, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 2, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 3, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 4, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 5, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 6, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 7, threadId.y: 5, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 0, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 1, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 2, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 3, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 4, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 5, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 6, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 7, threadId.y: 6, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 0, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 1, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 2, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 3, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 4, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 5, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 6, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 7, threadId.y: 7, threadId.z: 0) execution time for copying data from GMEM to SMEM: 2872 clock cycle
(threadId.x: 0, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 1, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 2, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 3, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 4, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 5, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 6, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 7, threadId.y: 0, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 0, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 1, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 2, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 3, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 4, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 5, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 6, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 7, threadId.y: 1, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 0, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 1, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 2, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 3, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 4, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 5, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 6, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 7, threadId.y: 2, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 0, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 1, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 2, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 3, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 4, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 5, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 6, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 7, threadId.y: 3, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2857 clock cycle
(threadId.x: 0, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 1, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 2, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 3, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 4, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 5, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 6, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 7, threadId.y: 4, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 0, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 1, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 2, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 3, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 4, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 5, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 6, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 7, threadId.y: 5, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 0, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 1, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 2, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 3, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 4, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 5, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 6, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 7, threadId.y: 6, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 0, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 1, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 2, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 3, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 4, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 5, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 6, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
(threadId.x: 7, threadId.y: 7, threadId.z: 1) execution time for copying data from GMEM to SMEM: 2862 clock cycle
```

## Conclusion

In this blog, Microbenchmarking was used to show that threads are grouped in warp, based on consecutive threadId of the threads. 
 
## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [Demystifying GPU Microarchitecture through Microbenchmarking](https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf)
