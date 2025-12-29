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

### Forward propagation steps in a three layer neural network.
{% include image-gallery.html images="forward_propagation.png" height="400" %} 
<br> 
 
### First layer

### Second layer

### Third layer

## Back propagation


## Conclusion


## References

 
