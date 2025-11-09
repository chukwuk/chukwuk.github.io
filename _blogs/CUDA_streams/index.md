---
layout: post
title: Application of CUDA streams for computing summation of 2D array along the rows.
description: Runtime comparison of asynchronous and synchronous data transfer for reduction kernel.
skills: 
  - CUDA Programming.
  - C++/C Programming.
  - CUDA Streams.
  - Asynchronous and synchronous data transfer. 

---

---
## Introduction  
CUDA streams is an advanced feature of CUDA programming toolkit that allows the overlap of data transfer with kernel execution. A stream in CUDA is a command sequence issued on the device by the host code, in which the command sequence execute on device in the order it was issued by the host code. Command sequence within a stream are guaranteed to execute in the specified order but command sequence in different streams can be interleaved and, when possible, they can even run concurrently. 
In this technical blog post, I will discuss how to apply CUDA streams in CUDA kernel computation for the summation of 2D array along the rows. Reduction of 2D/3D array along an axis is a common computational operation in deep learning. The code used in this blog was ran on Nvidia RTX 5070 Ti.  

## CUDA Execution Flow

There are two CUDA execution flow, which are the serial execution flow (default stream) and concurrent execution flow (non-default stream). Figure 1 shows the different CUDA execution flow on NVIDIA RTX 5070 Ti. To learn more about the different CUDA execution model, please visit [Mark Harris blog post](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) 

### Execution flow for default(single) stream and non-default(mutiple) stream on Nvidia RTX 5070 Ti
{% include image-gallery.html images="Asynchronous_transfer_4.png" height="400" %} 

<br>

<br>

## Kernel implemention for 2D array summation along the row for default stream and non-default stream

For the default stream kernel implementation, each GPU thread sums all the element in the same row. Row major storage is used for the 2D array storage in GPU memory. The kernel implementation can also be used for 2D array summation along the column but the column major storage should be used to store the 2D array in the GPU memory and the variable nCols (number of columns) will be replaced with number of rows. The default stream kernel implementation will not work for the non-default stream because the kernel execution in each stream will be operate on the same section of the data. In the non-default stream, the kernel execution in different streams should operate on different section of the 2D array data. Therefore, in the kernel implementation for the non-default stream, the offset variable is used to calculate the section of the 2D array data for the operation by the kernel execution in the different streams. The offset variable is different for each stream, please check out my [Github repo](https://github.com/chukwuk/CUDA_streams) for more information on how the offset is calculated.         

```cuda
// A simple CUDA kernel to reduce 2D array along the row for default(single) stream.

__global__  void reductionSum(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols) {

   int gid = (blockIdx.x *  blockDim.x +  threadIdx.x);
   size_t shift = (size_t)gid *  (size_t) nCols;
   if (gid < numData) {
      int sum = 0;
      for (size_t i = 0; i < nCols; i++) {
          sum += reduceData[shift + i];
      }
      sumData[gid] = sum;
   }
}
```


```cuda
// A simple CUDA kernel to reduce 2D array along the row for non-default(multiple) stream. 

__global__  void reductionSum(int* reduceData, int* sumData, unsigned long int numData, unsigned int nCols, int offset ) {

   int gid = offset + (blockIdx.x *  blockDim.x +  threadIdx.x);
   size_t shift = (size_t)gid *  (size_t) nCols;
   if (gid < (offset + numData)) {
      int sum = 0;
      for (size_t i = 0; i < nCols; i++) {
          sum += reduceData[shift + i];
      }
      sumData[gid] = sum;
   }
}
```


## References
* [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
* [CUDA Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf) 


