---
layout: post
title: Application of CUDA streams for computing reduction of 2D array along the rows.
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
In this technical blog post, I will discuss how to apply CUDA streams in CUDA kernel computation for the reduction of 2D array along the rows. Reduction of 2D/3D array along an axis is a common computational operation in deep learning. The code used in this blog was ran on Nvidia RTX 5070 Ti.  

### Embeed images
{% include image-gallery.html images="Asynchronous_transfer_4.png" height="400" %} 

<br>

<br>


## References
* [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
* [CUDA Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf) 


