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
# Introduction

Euclidean matrix distance is an nxn matrix representing the euclidean distance between set of n points in Euclidean space. Euclidean distance matrix has so many application in machine learning, Engineering and robotics and image processing. This technical blog will discuss step by step on how to optimize CUDA kernel for Euclidean Distance Matrix for 2D coordinate points. 

# Memory bound vs Compute bound for Euclidean distance Matrix

The eucildean distance calculation between 2D cooordinate points involves two substraction, two multiplication, one addition and one sqrt. According to this [post](https://forums.developer.nvidia.com/t/performance-tweak-for-single-precision-square-root/173274), sqrt() function involves about four to five floating point operation. Therefore, one euclidean distance calculation between 2D coordinates points involves 9 floating point operation. For 20224 2D coordinate points, which is what was used in this optimization. 

   1. Total FLOPS: 20224<sup>2</sup> * 9 FLOPS = 3.68GFLOPS.
   2. Minimum total data to read: 20224 * 8B = 161792B(162KB)
   3. Total data to write: 20224<sup>2</sup> * 4B = 1.64GB
   
The Nvidia RTX 5070 Ti has a memory bandwith of 896GB/sec and has a performance of 133TFLOPS. Therefore, the theoretical time for the calculation is 27.69 milliseconds while the theoretical total time for data read and write is 5.48 milliseconds assuming the both total read and write is 4.91GB. This reason for the approximation is that each GPU thread of 20224 threads reads 162 KB from global memory wihout caching. This theoretical calculation shows that the euclidean distance matrix calculation is compute-bound.       

# Naive kernel implementation
 

# References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
