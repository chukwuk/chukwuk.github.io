---
layout: post
title: Optimizing FP32 Matrix Multiplication on NVIDIA GPU, that achieves 93 percent of CUBLAS performance based on Nsight Compute Analysis.
description: FP32 Matrix multiplication optimization that achieves 93 percent of CUBLAS performance.
skills: 
  - CUDA Programming.
  - C++/C Programming.
  - Shared and Global memory.
  - CUDA Warps 
  - Asynchronous data transfer. 
  - Warp Specialization.
---

---
## Introduction.

Matrix multiplication is an important algorithm used in training and inference of large deep neural network models, in which the matrix multiplication algorithm for the large deep neural networks models run on GPU hardware because parallel processing of the output data. For the deep neural network models to be faster, the matrix multiplication algorithm should run efficiently on the GPU hardware. In this technical blog, I will discuss step by step on how to optimize matrix multiplication algorithm on NVIDIA GPU (RTX 5070 Ti) to get 93% performance of cuBLAS. cuBLAS is optimized NVIDIA library for basic linear algebra. The matrix size (M=10240,K=4096,N=4096) will be used for the step by step optimization discussion.  

| Matrix Size (MxKxN) | fp32 Custom GEMM Kernel | cuBLAS  | Speedup |
| ------------------- | ----------------------- | ------- | ------- |
| 4096x4096x4096    | 26.77 TFLOPS  | 30.07 TFLOPS | 0.89x |
| 10240x4096x4096   | 28.22 TFLOPS  | 30.70 TFLOPS | 0.92x |
| 16384x16384x16384 | 29.47 TFLOPS  | 31.80 TFLOPS | 0.93x |

## Memory bound vs Compute bound for Matrix Multiplication.


The matrix mutiplication to produce one output data involves element wise multiplication of A rows with B column and then addition of the products. Therefore, for matrix size (M=10240, K=4096, N=4096), the number of flops required for one output data is (4096 * 2) FLOP. 

   1. Total FLOPS: 2 * 4096 * 4096 * 10240 FLOP = (0.34 TFLOP).
   2. Minimum total data to read: 10240 * 4096 * 4B + 4096 * 4096 * 4B +  4096 * 4096 * 4B = 301989888B(302 MB).
   3. Total data to write: 10240 * 4096 * 4B = 167772160B(168 MB).
   
The Nvidia RTX 5070 Ti has a memory bandwith of 896GB/sec and has a fp32 compute throughput of 41 TFLOPS. Therefore, the theoretical time for the calculation is 8.29 milliseconds while the theoretical total time for data read and write is 0.54 milliseconds assuming the both total read and write is 470 MB. This simple theoretical calculation shows that the matrix multiplication is compute-bound.       


## Kernel 1: Naive implementation.

## Kernel 2: .
## Kernel 3: Shared Memory Cache-Blocking.



## Kernel 4: .

## Kernel 5: . 

## Kernel 6: .

## Summary



## Conclusion


This technical blog discussed step by step on how to optimize kernel optimization for matrix multiplication of NVIDIA GPU. 


## References

* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [SOL Analysis with NVIDIA Nsight Compute](https://www.youtube.com/watch?v=uHN5fpfu8As)
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
* [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
