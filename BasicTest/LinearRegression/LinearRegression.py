# import package
import sys
import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# set files' root and seed
root = r"F:\Jzmo\tf\BasicTest\LinearRegression"
seed = np.random.seed(21)
# set sesstion
sess = tf.InteractiveSession()
# set learning parameter
learning_rate = 0.1
learning_epochs = 200
display_time = 10
# set train dataset
train_x = np.arange(0,10,1,dtype = np.float32)
train_y = train_x*2+np.random.randn(10,)
print("train Y:", train_x)
print("train Y:", train_y)

# set test dataset
test_x = np.asarray([1.3,4.6,6.6,3.4,9.4,4.2,1.1,2.3,4.9])
test_y = test_x*2+np.random.randn(9,)
print("test X:", test_x)
print("test Y:", test_y)

# set placeholder
n_samples = train_x.shape[0]
input_x = tf.placeholder(dtype = np.float32,name = 'input_x')
input_y = tf.placeholder(dtype = np.float32,name = 'input_x')

# set weight and bias
Weight = tf.Variable(np.random.random(),name = 'Weight')
bias = tf.Variable(np.random.random(),name = 'bias')

# set regression function
perd = tf.add((input_x*Weight),bias)

# set cost and optimizer
cost = tf.reduce_sum(tf.pow(perd-input_y,2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# set test data base
test_input_y = tf.placeholder(dtype = np.float32,name = 'test_input_y')
test_input_x = tf.placeholder(dtype = np.float32,name = 'test_input_x')
pred_test_y = tf.add(Weight*test_x,bias)
test_loss = tf.reduce_sum(tf.pow(pred_test_y-test_y,2))/(2*9)

# init saver
saver = tf.train.Saver()
# init variable
init = tf.global_variables_initializer()

## init summary writer
tf.summary.scalar('test_loss',test_loss)
merged_summary_op = tf.summary.merge_all()

summary_wirter = tf.summary.FileWriter(root+'/logs',sess.graph)

sess.run(init)

for epoch in range(learning_epochs):
    for x,y in zip(train_x,train_y):
        sess.run(optimizer,feed_dict = {input_x:x,input_y:y})
        
    # log information at each eposh
    if epoch % display_time == 0 :
        print("training eposh:",epoch," Weight:",sess.run(Weight),
            " bias:",sess.run(bias),
            " cost: ",sess.run(test_loss,feed_dict = {
            input_x:x,input_y:y,
            test_input_x:test_x,test_input_y:test_y}))
        summary_str = sess.run(merged_summary_op)
        summary_wirter.add_summary(summary_str,epoch*display_time)
print("=================================")
print("training eposh:",epoch," Weight:",sess.run(Weight),
    " bias:",sess.run(bias))
# testing and print accurancy
print("test square loss:",sess.run(test_loss,feed_dict = {
    input_x:x,input_y:y,
    test_input_x:test_x,test_input_y:test_y}))
W = sess.run(Weight)
b = sess.run(bias)
# save net
savepath = root + "\linearmodel.ckpt"
savepath = saver.save(sess,savepath)
    
#plot result
# plt.plot(train_x,train_y,'ro',label='Original data')
# plt.plot(train_x,W*train_x+b,label = 'Fitted line')
# plt.plot(test_x,test_y,'bo',label = 'Test data')
# plt.legend()
# plt.show()    
