This program is an one layer simple neural network to identify handwritten digits
Compare different learning rate and batch size
### data 
the [MNIST](http://yann.lecun.com/exdb/mnist/) DATABASE of handwritten digits is a classic problem in machine learning. The problem is to identify the grayscale handwritten digital image of 28x28 pixels as the corresponding number, which ranges from 0 to 9.  
![image](http://www.tensorfly.cn/tfdoc/images/mnist_digits.png)

download and extracte data by [input_data.py](https://tensorflow.googlesource.com/tensorflow//master/tensorflow/examples/tutorials/mnist/input_data.py#)

* train data  
	* 55000 grayscale images
	* including images 1 * 784 (28 * 28) *  55000 and labels (10 * 55000)
	* to train net
* test data
	* 10000 grayscale images
	* including images 1 * 784 (28 * 28) *  10000 and labels (10 * 10000)
	* to get the accuracy during iterative training process
* valid data
	* 5000 grayscale images
	* including images 1 * 784 (28 * 28) * 5000 and labels (10 * 5000)
	* to get the final accuracy
### process

### parameter
learning rate  | batch size | epochs | activation functions | optimizer | loss function  
:--------- | :--------| :-------- | :-------- | :-------- | :-------- 
0.1 - 0.001  | 200 | 20  | Softmax | Gradient Descent | Cross-entropy
### result
