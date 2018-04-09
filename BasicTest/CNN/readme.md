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
![image](https://github.com/Jzmo/tf/raw/master/BasicTest/CNN/cnn1.png)

### parameter
learning rate  | batch size | epochs | activation in conv| pooling |optimizer | activation in fc| loss function  
:--------- | :--------| :-------- | :-------- | :-------- | :--------  | :--------  | :-------- 
0.0001  | 200 | 20  | Relu | max pool | Adam | Softmax | Cross-entropy

### methods
**Relu**  
In the context of artificial neural networks, the rectifier is an activation function defined as the positive part of its argument:
![equation](http://latex.codecogs.com/gif.latex?f(x)=x^{&plus;}=\max(0,x))

**Adam** 

### result
