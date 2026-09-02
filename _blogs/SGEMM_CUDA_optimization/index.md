---
layout: post
title: Optimizing FP32 Matrix Multiplication on NVIDIA GPU, that achieves 93 percent of CUBLAS performnce based on Nsight Compute Analysis.
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


| Matrix Size (MxKxN) | fp32 Custom GEMM Kernel | cuBLAS  | Speedup |
| ------------------- | ----------------------- | ------- | ------- |
| 4096x4096x4096    | 26.77 GFLOPS  | 30.07 GFLOPS | 0.89x |
| 10240x4096x4096   | 28.22 GFLOPS  | 30.70 GFLOPS | 0.92x |
| 16384x16384x16384 | 29.47 GFLOPS  | 31.80 GFLOPS | 0.93x |

## Memory bound vs Compute bound for Matrix Multiplication.

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
