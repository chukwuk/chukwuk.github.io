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

Neural Network is a type of computational model that is inspired by the human brain, in which the neurons are organized in layers and the neurons in different layer are interconnected. Neural networks has many application in autonomous driving, natural language processing and image/speech recognition. In this technical blog, I will discuss CUDA implementation of one forward and backward propagation of a three-layer neural network, in which the result of the CUDA implementation was compared with Pytorch implementation. The comparison with Pytorch implementation helps with debugging since Pytorch has been well tested by the deep learning development community. The Pytorch initialized weights and biases are used in CUDA implementation because it easier for comparison between Pytorch and CUDA implementation. Additionally, the kernel functions used in this blog were not optimized. I will discuss kernel optimization in a another blog and also there is a lot of optimization opportunities for all the kernel functions used here. 

## Forward Propagation

Forward propagation is a process where the neural network takes an input to produce an output or prediction. In the three layer neural network, the two hidden layer has a ReLu(Rectified Linear Unit) activation and output layer has a sigmoid activation function. 

### Forward propagation steps in a three layer neural network(The steps with the same color happen in the same layer).
{% include image-gallery.html images="forward_propagation.png" height="400" %} 
<br> 
 
### First Layer (First step).

The first step involves matrix multiplication between First layer weights and input data and then the addition of the First Layer bias. Row major storage is used to weight and bias in GPU memory while column major storage is used to store the input data in GPU memory. The column major storage is used to store the matrix mutiplication product in GPU memory. The first step result is left in the GPU memory to be used by the second step. Please note, different memory should be allocated for the weight and bias for one layer rather one memory for weight and bias of one layer as in this blog.   
    
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

Third step operation is the same as the first step but it involves matrix mutiplication of second layer weights with the output of layer one (second step) and then addition of the second layer bias. The third step uses the same kernel function as the first step. 
### matrix representation of the third step.
{% include image-gallery.html images="step_3_matrix_multiply.png" height="400" %} 
<br>  
### Second layer (four step).

Fourth step operation is the same as the second step, in which ReLu function is applied on the third step result. The fourth step uses the same kernel function as second step.  
### matrix representation of the fourth step.
{% include image-gallery.html images="step_4_ReLu_function.png" height="400" %} 
<br>  
### Third layer (fifth step).

Fifth step operation is the same as the first step but it involves matrix mutiplication of third layer weights with output layer two(fourth step) and then addition of the third layer bias. The fifth step uses the same kernel function as the first step. The last layer must have a single neuron since the code is structured for only binary classification. In the future, the code will be updated to multiclass classification with softmax. 
### matrix representation of the fifth step.
{% include image-gallery.html images="step_5_matrix_multiply.png" height="400" %} 
<br>
### Third layer (sixth step).

The sixth step involves applying the sigmoid function on the product of the fifth step.
### matrix representation of the sixth step.
{% include image-gallery.html images="step_6_sigmoid_function.png" height="400" %} 
```cuda
// Kernel function for the sixth step
__global__  void matrixSigmoid(float* activation, int actLength) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;

    if (gid < actLength) {
       activation[gid] = 1/(1 + exp(-activation[gid]));
    }
}
```
## Backward propagation.

Backward propagation involves changing the weight and biases through the network in other to minimize the error. The backward propagation is more computationally complex than the forward propagation because it involves calculation of the gradient of the weights and biases by applying chain rules backward through the neural network. In the technical blog, the cuda implementation of the backward propagation was simplified by starting from the sixth step. 
### Sixth step
The sixth step involves substracting the actual data from the predicted data (dL/dZ3).  
{% include image-gallery.html images="backward_propagation_1.png" height="400" %} 
```cuda
// kernel function for the sixth step
__global__  void elementWiseSub(float* firstArray, float* secondArray, int arraySize) {

   
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (gid < (arraySize)) {
	firstArray[gid]=firstArray[gid] - secondArray[gid]; 
   }

}
```
<br>
### Fifth step

The fifth step requires calculating the derivatives of W3, b3 and a2 with respect with to the loss function. First, dL/dZ3 is transposed from column major storage to row major storage to make easier to reuse kernel function.   
```cuda
// kernel function used for transpose dL/dZ3 from column major storage to row major storage
__global__  void matrixTransposeSubBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    int index = gid / ncols;
    int indexR = gid - (index * ncols);

    if (gid < (nrows*ncols)) {
       matrixArrayTranspose[index+(indexR*nrows)] = matrixArray[gid];
    } 
}
```
```cuda
// kernel function used for transpose a2 from column major storage to row major storage.
// kernel function add one extra rows that is filled with one b/cos it used to calculate dL/db3 since W and b is in the same matrix 
__global__  void matrixTransposeAddBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    int index = gid / ncols;
    int indexR = gid - (index * ncols);

    if (gid < (nrows*(ncols+1))) {
       if (gid < (nrows*ncols)) { 
          matrixArrayTranspose[index+(indexR*nrows)] = matrixArray[gid];
       } else {

          matrixArrayTranspose[gid] = 1.0;
       }
       
    } 
}
```
```cuda
// kernel function for calculation of dL/dW3 and dL/db3 
__global__  void matrixdL_dW3(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x; 
    if (gid < (wRows * xCols)) {
        	
        int index = gid / xCols;
        int indexW = index * (wColsXRows);
        int indexStart = gid % xCols;
        int IndexMul = indexStart * wColsXRows; 
    	float sum = 0.0;
    	for (int i = 0; i < wColsXRows; i++)  {
	       sum+=(weightBias[i+indexW] * xData[i+IndexMul]);   
       	}
        activationValues[gid] = sum/float(wColsXRows);	
	 
    }

}
```
{% include image-gallery.html images="backward_propagation_2.png" height="400" %} 
<br>
{% include image-gallery.html images="backward_propagation_3.png" height="400" %} 
<br>

## Conclusion

This techical blog discussed step by step CUDA implementation of a one forward and backpropagation of a three layer neural network and compare results with Pytorch. All my code is available on (Github)[https://github.com/chukwuk/CUDA_implementation_of_a_three_layer_neural_network/tree/main].

## References

* (Neural Networks and Deep Learning cousera course)[https://www.coursera.org].
* (Google)[https://www.google.com/?zx=1767253382136&no_sw_cr=1].
 
