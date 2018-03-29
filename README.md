# some project based on tensorflow
### A simple linear regression
This project just use to understand the basic process of tensorflow learning<br>
.py file: Jzmo/tf/BasicTest/Linear Regression<br>

#### data
>train data:<br>
>>x:[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]<br>
>>y:[-0.05196425  1.88880395  5.0417968   4.74326071  8.74538768  8.28894624  11.79413562 13.76542871 17.12814404 17.98737405]<br>
>test data:<br>
>>x:[1.3 4.6 6.6 3.4 9.4 4.2 1.1 2.3 4.9]<br>
>>y:[ 2.6  9.2 13.2  6.8 18.8  8.4  2.2  4.6  9.8]<br>
#### process
* set parameter
  * learning_epochs -- like the number of iterations
  * learning_rate -- rate of descent
* set train and test dataset (and valid dataset)
* set variable of training net
  * the input of x and y (set with placeholder)
  * the wight and bias
  * the function to predict y(pred_y = x*w+b)
  * the cost and optimizer
* run session to train data
  * init all variable
  * run optimizer
  * if needed, run other operation and print log information
* test the accurancy
* plot (use matplotlib)
#### result
![](https://github.com/Jzmo/tf/BasicTest/LinearRegression/linearRegression.png)

### notMNIST from Udacity
