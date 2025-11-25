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
(ThreadId.x: 0) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 1) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 2) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 3) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 4) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 5) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 6) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 7) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 8) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 9) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 10) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 11) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 12) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 13) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 14) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 15) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 16) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 17) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 18) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 19) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 20) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 21) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 22) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 23) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 24) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 25) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 26) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 27) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 28) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 29) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 30) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 31) execution time for copying data from GMEM to SMEM: 2534 clock cycle
(ThreadId.x: 32) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 33) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 34) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 35) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 36) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 37) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 38) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 39) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 40) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 41) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 42) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 43) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 44) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 45) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 46) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 47) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 48) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 49) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 50) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 51) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 52) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 53) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 54) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 55) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 56) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 57) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 58) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 59) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 60) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 61) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 62) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 63) execution time for copying data from GMEM to SMEM: 2544 clock cycle
(ThreadId.x: 64) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 65) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 66) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 67) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 68) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 69) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 70) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 71) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 72) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 73) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 74) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 75) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 76) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 77) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 78) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 79) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 80) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 81) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 82) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 83) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 84) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 85) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 86) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 87) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 88) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 89) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 90) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 91) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 92) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 93) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 94) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 95) execution time for copying data from GMEM to SMEM: 2537 clock cycle
(ThreadId.x: 96) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 97) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 98) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 99) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 100) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 101) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 102) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 103) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 104) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 105) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 106) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 107) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 108) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 109) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 110) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 111) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 112) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 113) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 114) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 115) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 116) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 117) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 118) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 119) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 120) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 121) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 122) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 123) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 124) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 125) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 126) execution time for copying data from GMEM to SMEM: 2541 clock cycle
(ThreadId.x: 127) execution time for copying data from GMEM to SMEM: 2541 clock cycle
```

## Warp in 2D block

In the 2D block, the blockDim.x is 16, blockDim.y is 8 and blockDim.z is 1. Warp one has  ℤ<sup>2</sup>(D=\{(threadId.x,threadId.y)\in \mathbb{Z}^{2}\mid 0&le;x&le;16,0&le;y&le;1\}\), Warp two has threadId.x ∈ {32,33,34.....63},

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
(ThreadId.x: 0, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 1, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 2, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 3, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 4, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 5, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 6, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 7, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 8, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 9, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 10, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 11, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 12, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 13, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 14, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 15, ThreadId.y: 0) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 0, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 1, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 2, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 3, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 4, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 5, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 6, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 7, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 8, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 9, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 10, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 11, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 12, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 13, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 14, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 15, ThreadId.y: 1) execution time for copying data from GMEM to SMEM: 2864 clock cycle
(ThreadId.x: 0, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 1, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 2, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 3, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 4, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 5, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 6, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 7, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 8, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 9, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 10, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 11, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 12, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 13, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 14, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 15, ThreadId.y: 2) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 0, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 1, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 2, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 3, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 4, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 5, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 6, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 7, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 8, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 9, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 10, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 11, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 12, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 13, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 14, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 15, ThreadId.y: 3) execution time for copying data from GMEM to SMEM: 2865 clock cycle
(ThreadId.x: 0, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 1, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 2, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 3, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 4, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 5, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 6, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 7, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 8, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 9, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 10, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 11, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 12, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 13, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 14, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 15, ThreadId.y: 4) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 0, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 1, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 2, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 3, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 4, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 5, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 6, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 7, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 8, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 9, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 10, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 11, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 12, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 13, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 14, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 15, ThreadId.y: 5) execution time for copying data from GMEM to SMEM: 2951 clock cycle
(ThreadId.x: 0, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 1, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 2, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 3, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 4, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 5, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 6, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 7, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 8, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 9, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 10, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 11, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 12, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 13, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 14, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 15, ThreadId.y: 6) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 0, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 1, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 2, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 3, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 4, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 5, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 6, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 7, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 8, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 9, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 10, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 11, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 12, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 13, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 14, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
(ThreadId.x: 15, ThreadId.y: 7) execution time for copying data from GMEM to SMEM: 2981 clock cycle
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
(ThreadId.x: 0, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 1, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 2, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 3, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 4, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 5, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 6, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 7, ThreadId.y: 0, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 0, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 1, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 2, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 3, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 4, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 5, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 6, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 7, ThreadId.y: 1, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 0, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 1, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 2, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 3, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 4, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 5, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 6, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 7, ThreadId.y: 2, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 0, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 1, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 2, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 3, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 4, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 5, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 6, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 7, ThreadId.y: 3, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2955 clock cycle
(ThreadId.x: 0, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 1, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 2, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 3, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 4, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 5, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 6, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 7, ThreadId.y: 4, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 0, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 1, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 2, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 3, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 4, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 5, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 6, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 7, ThreadId.y: 5, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 0, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 1, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 2, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 3, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 4, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 5, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 6, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 7, ThreadId.y: 6, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 0, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 1, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 2, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 3, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 4, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 5, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 6, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 7, ThreadId.y: 7, ThreadId.z: 0) execution time for copying data from GMEM to SMEM: 2961 clock cycle
(ThreadId.x: 0, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 1, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 2, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 3, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 4, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 5, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 6, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 7, ThreadId.y: 0, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 0, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 1, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 2, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 3, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 4, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 5, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 6, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 7, ThreadId.y: 1, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 0, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 1, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 2, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 3, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 4, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 5, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 6, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 7, ThreadId.y: 2, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 0, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 1, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 2, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 3, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 4, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 5, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 6, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 7, ThreadId.y: 3, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2866 clock cycle
(ThreadId.x: 0, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 1, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 2, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 3, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 4, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 5, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 6, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 7, ThreadId.y: 4, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 0, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 1, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 2, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 3, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 4, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 5, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 6, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 7, ThreadId.y: 5, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 0, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 1, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 2, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 3, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 4, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 5, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 6, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 7, ThreadId.y: 6, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 0, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 1, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 2, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 3, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 4, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 5, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 6, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
(ThreadId.x: 7, ThreadId.y: 7, ThreadId.z: 1) execution time for copying data from GMEM to SMEM: 2868 clock cycle
```

## Conclusion

In this blog, Microbenchmarking was used to show that threads are grouped in warp, based on consecutive threadId of the threads. 
 
## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [Demystifying GPU Microarchitecture through Microbenchmarking](https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf)
