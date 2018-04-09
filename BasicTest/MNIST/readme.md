This program is an one layer simple neural network to identify handwritten digits
Compare different learning rate and batch size
### data 
the [MNIST](http://yann.lecun.com/exdb/mnist/) DATABASE of handwritten digits is a classic problem in machine learning. The problem is to identify the grayscale handwritten digital image of 28x28 pixels as the corresponding number, which ranges from 0 to 9.  
![image](http://www.tensorfly.cn/tfdoc/images/mnist_digits.png)

* train data  
	* 55000 grayscale images
	* 1 * (28 * 28) *  55000
* test data
	* 10000 grayscale images
	* 1 * (28 * 28) *  10000
* valid data
	* 5000 grayscale images
	* 1 * (28 * 28) *  5000

### process
### parameter
learning rate  | batch size | epochs | activation functions | optimizer | loss function  
:--------- | :--------| :-------- | :-------- | :-------- | :-------- 
0.1 - 0.001  | 200 | 20  | Softmax | Gradient Descent | cross-entropy

#### result
