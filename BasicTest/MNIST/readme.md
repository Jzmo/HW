## MNIST
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
![image](https://github.com/Jzmo/tf/raw/master/BasicTest/MNIST/net1.png)

### parameter
learning rate  | batch size | epochs | activation functions | optimizer | loss function  
:--------- | :--------| :-------- | :-------- | :-------- | :-------- 
0.1 - 0.001  | 200 | 20  | Softmax | Gradient Descent | Cross-entropy

### methods
**softmax**  
The softmax function is used in various multiclass classification methods. In multinomial logistic regression and linear discriminant analysis, the input to the function is the result of K distinct linear functions, and the predicted probability for the j'th class given a sample vector x and a weighting vector w is:  
* ![equation](http://latex.codecogs.com/gif.latex?P(y=j|x)=\frac{e^{x^{T}w_{j}}}{\sum&space;_{k=1}^{K}e^{x^{T}w_{j}}})

**Cross entropy**  
Cross entropy can be used to define the loss function in machine learning and optimization. The true probability **p_i** is the true label, and the given distribution **q_i** is the predicted value of the current model.  Cross entropy is used to get a measure of dissimilarity between p and q:
![equation](http://latex.codecogs.com/gif.latex?H(p,q)=-\sum_{i}&space;p_{i}log(q_{i}))

### result
