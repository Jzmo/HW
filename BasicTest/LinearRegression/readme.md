### A simple linear regression
This project is to understand the basic process of tensorflow machine learning<br>

### data
* train data:<br>
  * x:[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]<br>
  * y:[-0.05196425  1.88880395  5.0417968   4.74326071  8.74538768  8.28894624  11.79413562 13.76542871 17.12814404 17.98737405]<br>
* test data:<br>
  * x:[1.3 4.6 6.6 3.4 9.4 4.2 1.1 2.3 4.9]<br>
  * y:[ 2.6  9.2 13.2  6.8 18.8  8.4  2.2  4.6  9.8]<br>
  
### process
* set parameter<br>
  * learning_epochs -- like the number of iterations<br>
  * learning_rate -- rate of descent<br>
* set train and test dataset (and valid dataset)<br>
* set variable of training net<br>
  * the input of x and y (set with placeholder)<br>
  * the wight and bias<br>
  * the function to predict y(pred_y = x*w+b)<br>
  * the cost and optimizer<br>
* run session to train data<br>
  * init all variable<br>
  * run optimizer<br>
  * if needed, run other operation and print log information<br>
* test the accurancy<br>
* plot (use matplotlib)<br>

### parameter
learning rate  |epochs | optimizer | loss function  
:--------- | :--------| :-------- | :-------- 
0.1  | 500 | Gradient Descent | Square Loss

### methods
**Gradient Descent**  
To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point. If instead one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent.
[see gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)  

![equation](http://latex.codecogs.com/gif.latex?x{_{n&plus;1}}=x_{n}-\gamma&space;\bigtriangledown&space;F(a^{{n}}))  

**Square Loss**    
Quadratic loss function is commonly use in linear regression. The loss function is:  

![equation](http://latex.codecogs.com/gif.latex?l(x)=C(x_{predict}-x_{input})^{2})  

### result
![image](https://github.com/Jzmo/tf/raw/master/BasicTest/LinearRegression/result.PNG)
![image](https://github.com/Jzmo/tf/raw/master/BasicTest/LinearRegression/linearRegression.png)
