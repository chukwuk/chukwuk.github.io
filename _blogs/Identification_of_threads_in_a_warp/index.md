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

A CUDA warp is a group of 32 threads that executes the same instruction. Understanding how threads are grouped in a warp is important for warp tiling used in optimizing matrix mutiplication, which is a common computation in deep learning. In this technical blog, I will discuss how the 32 threads are grouped as a warp. This [blog]( https://siboehm.com/articles/22/CUDA-MMM) discussed grouping of threads in a warp. In this technical blog, I will be microbenchmarking NVIDIA RTX 5070 Ti to identify which group of threads belong to a warp. This [paper](https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf) has more information about GPU microbenchmarking.

## Microbenchmarking 

CUDA programming model and NVIDIA hardware architecture never allows two different warps to access the shared memory simultaneously in a single clock cycle. Consequently, the number of clock cycles was measured for loading of data from the global memory(gmem) to shared memory(smem), since when one warp is accessing the shared memory, the other warp will be stalled and this will lead to different number of clock cycles for loading the data from the gmem to smem. The NVIDIA RTX 5070 ti has 128 cores per SM, so 128 blocksize was used to ensure the four warp can be executed in a single clock cycle, which will lead to stalling of the other three warps when loading data from gmem to smem. Threads in the same warp will have the same number of clock cycles for loading the data from gmem to smem, which is used to identify threads in the same warp. The struct below was used to collect information about the number of clock cycles, threadId.x, threadId.y and threadId.z each of thread. 

```cuda
struct threadProperties {
    unsigned long long time;
    int thread_x;
    int thread_y;
    int thread_z;
    int value;
};
```

## Warp in ID block



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

## Warp in 2D block


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


## Warp in 3D block


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

## Conclusion
 
## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [Demystifying GPU Microarchitecture through Microbenchmarking](https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf)
