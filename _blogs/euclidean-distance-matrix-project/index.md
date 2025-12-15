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

The naive implementation involves each GPU thread computing the euclidean distance between one coordinate and every other coordinate, which means each GPU thread will compute n euclidean distance out of n x n euclidean distance matrix. In CUDA programming model, threads are execute by warp, which is a group of 32 threads that executes the same instruction simultaneously(Single Instruction Multiple Threads). The minimum memory transaction by a warp is 32 bytes, which means a warp need four memory transaction to write 32 float data type if the data are adjacent. In the case, where the data is not aligned, the warp will need 32 memory transaction(1024bytes) to service the memory access, which is a waste of 992 bytes. 

## Kernel 2: Global Memory Coalescing
 

## Kernel 3: Shared Memory Cache-Blocking

## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [SOL Analysis with NVIDIA Nsight Compute](https://www.youtube.com/watch?v=uHN5fpfu8As)
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
