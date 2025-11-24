---
layout: post
title: How CUDA threads are grouped in a Warp.
description: Group of threads in a Warp was identified was measuring number of clock cycles for loading data from Global memory to Shared memory.
skills: 
  - CUDA Programming.
  - C++/C Programming.
  - Shared and Global memory.
  - CUDA Warps 

---

---
## Introduction  

A CUDA warp is a group of 32 threads that executes the same instruction. Understanding our threads are grouped in a warp is important for warp tiling used in optimizing matrix mutiplication. In this technical blog, I will discuss how the 32 threads are grouped as a warp. This blog discussed how the formul;a for identifying threads in a warp but here I will be microbenching NVIDIA RTX 5070 to confirm the grouping of threads in a warp.   
## Warp in ID block


```cuda
// A simple CUDA kernel.

```

## Warp in 2D block

## Warp in 3D block

## Conclusion



## Conclusion.

 
## References
* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

