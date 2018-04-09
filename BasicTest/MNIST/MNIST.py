# download and extrict the traning and testing files
import input_data
mnist = input_data.read_data_sets('./',one_hot = True)

# import other package
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import summaries
img_size = 28
n_output = 10
n_feature = img_size*img_size
traning_num = 55000
validation_num = 5000
test_num = 10000

root = r'D:\code\tf\BasicTest\MNIST'

def vector_2_image(img_vector, img_height, img_width):
    img = tf.reshape(img_vector,[-1,img_height,img_width,1])            
    return img
    
## plot one of the images and its label 
# img = vector_2_image(mnist.train.images[1000,:],img_size,img_size)
# plt.imshow(img)
# plt.title(mnist.train.labels[1000])
# plt.show()

# training data:
# mnist.train.images[:,:]
# mnist.train.labels[:,:]

## one layer simple neural network
learning_rate = 0.0001
learning_epochs = 20
batch_size = 20

learning_iterations = int(traning_num / batch_size)

#set placeholder
with tf.name_scope('input'):
    x = tf.placeholder(dtype = np.float64,name = 'x',shape = [None,n_feature])
    y_ = tf.placeholder(dtype = np.float64,name = 'y_',shape = [None,n_output])

#set variables:
with tf.name_scope('layer1'):
    W = tf.Variable(np.zeros([img_size*img_size,n_output]))
    b = tf.Variable(np.zeros(n_output,))
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    loss = -tf.reduce_sum(y_*tf.log(y))
    summaries.variable_summaries(W,'weight')
    summaries.variable_summaries(b,'bias')
    tf.summary.histogram('activated_output',y)
    tf.summary.scalar('cross_entropy',loss)

#accurancy
with tf.name_scope('output'):
    right_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accurancy = tf.reduce_mean(tf.cast(right_pred,np.float64))
    tf.summary.scalar('test_accurancy',accurancy)
    
# set summary
sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(root + '/train_log',sess.graph)    
test_writer = tf.summary.FileWriter(root + '/test_log')    

# set net choice
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)

for epoch in range(learning_epochs):
    for iteration  in range(learning_iterations):
        images_feed, labels_feed = mnist.train.next_batch(batch_size)
        summary_str,_ = sess.run([merged, optimizer],feed_dict = 
            {x:images_feed,y_:labels_feed})
        train_writer.add_summary(summary_str,iteration+epoch*learning_iterations)
    #log information
    images_test, labels_test = mnist.train.next_batch(test_num)
    summary_str, ac = sess.run([merged, accurancy],feed_dict = 
            {x:images_test,y_:labels_test})

    print("epoch:",epoch, "accurancy:",ac)
    
train_writer.close()
test_writer.close()

    