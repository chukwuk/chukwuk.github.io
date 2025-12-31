---
layout: post
title: CUDA Implementation of a three layer neural network. 
description: CUDA implementation of a three layer neural network was compared with Pytorch implementation. 
skills: 
  - CUDA programming.
  - C++/C programming.
  - Pytorch.
  - Python programming.
---

---
## Introduction  

Neural Network is a type of computational model that is inspired by the human brain, in which the neurons are organized in layers and the neurons in different layer are interconnected. Neural networks has many application in autonomous driving, natural language processing and image/speech recognition. In this technical blog, I will discuss CUDA implementation of one forward and backward propagation of a three-layer neural network, in which the result of the CUDA implementation was compared with Pytorch implementation. The comparison with Pytorch implementation helps with debugging since Pytorch has been well tested by the deep learning development community. Additionally, the kernel functions used in this blog were not optimized. I will discuss kernel optimization in a another blog and also there is a lot of optimization opportunities for all the kernel functions used here. 

## Forward Propagation

Forward propagation is a process where the neural network takes an input to produce an output or prediction. In the three layer neural network, the two hidden layer has a ReLu(Rectified Linear Unit) activation and output layer has a sigmoid activation function. 

### Forward propagation steps in a three layer neural network(The steps with the same color happen in the same layer).
{% include image-gallery.html images="forward_propagation.png" height="400" %} 
<br> 
 
### First Layer (First step).

The first step involves matrix multiplication between First layer weights and input data and then the addition of the First Layer bias. Row major storage is used to weight and bias in GPU memory while column major storage is used to store the input data in GPU memory because these will allow global coalescing. The column major storage is used to store the matrix mutiplication product in GPU memory because this will allow global coalescing in the third step. The first step result is left in the GPU memory to be used by the second step.   
### matrix representation of the first step
{% include image-gallery.html images="step_1_matrix_multiply.png" height="400" %} 
```cuda
// kernel function for the first step
__global__  void matrixMulAddRowBasedARR2(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    if (gid < (wRows * xCols)) {
        	
	int index = gid / wRows;
	int indexR = gid - (index * wRows);
        int indexW = index * (wColsXRows);
        int indexMul = indexR * (wColsXRows+1); 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexMul] * xData[i+indexW]);
           
       	}
	sum+=weightBias[indexMul+wColsXRows];
        activationValues[gid] = sum;	
    }

}
```
### First layer (second step).

The second step involves applying the ReLu function on the product of the first step.
### matrix representation of the second step.
{% include image-gallery.html images="step_2_ReLu_function.png" height="400" %} 
```cuda
// kernel function for the second step
__global__  void matrixReLu(float* activation, int actLength) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;

    if (gid < actLength) {
       activation[gid] = (activation[gid] > 0.0) ? activation[gid] : 0.0;
    }
}

```
### Second layer (Third step).

Third step operation is the same as the first step but it involves matrix mutiplication of second layer weights with the output of Layer one (second step) and then addition of the second layer bias. The third step uses the same kernel function as the first step. 
### matrix representation of the third step.
{% include image-gallery.html images="step_3_matrix_multiply.png" height="400" %} 
<br>  
### Second layer (four step).

Fourth step operation is the same as the second step, in which ReLu function is applied on the third step result. The fourth step uses the same kernel function as second step.  
### matrix representation of the fourth step.
{% include image-gallery.html images="step_4_ReLu_function.png" height="400" %} 
<br>  
### Third layer (fifth step).
### Third layer (sixth step).


## Back propagation


## Conclusion


## References

 
