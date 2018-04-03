# download and extrict the traning and testing files
import input_data
mnist = input_data.read_data_sets('./',one_hot = True)

# import other package
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

img_size = 28
traning_num = 55000
validation_num = 5000
test_num = 10000
batch_size = 200

learning_rate = 0.0001
learning_epochs = 20
learning_iterations = int(traning_num / batch_size)
# cnn
def conv2D(origin,filter):
    return tf.nn.conv2d(origin,filter,strides = [1,1,1,1],padding = 'SAME',)
    
def max_pool_2x2(origin):
    return tf.nn.max_pool(origin, ksize = [1,2,2,1],
        strides = [1,2,2,1],padding = 'SAME')
        
def weight_variable(shape):
    initial = tf.truncated_normal(
        shape = shape,
        stddev = 0.1,
        dtype = tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(
        0.1,
        shape = shape,)
    return tf.Variable(initial)
# interactive Session
sess = tf.InteractiveSession()

# np.random.seed(12)
in_height = img_size
in_width = img_size
in_channels = 1
filter_height = 5
filter_width = 5
######################## convolutional layer 1
out_channels_layer1 = 32

# convolutional layer 1
# placeholder x y_ h_conv1 h_pool1       
x = tf.placeholder(dtype = tf.float32,name = 'x',
    shape = [batch_size, in_height*in_width*in_channels])
y_ = tf.placeholder(dtype = tf.float32,name = 'y_')

# reshape input data
x_input = tf.reshape(x,[-1,img_size,img_size,1])

# Weight and bias
W_conv1 = tf.Variable(weight_variable([
    filter_height, filter_width, in_channels, out_channels_layer1]))
b_conv1 = tf.Variable(bias_variable([out_channels_layer1]))

# conv and pool 
h_conv1 = tf.nn.relu(conv2D(x_input,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
######################## concolutioanl layer 2
out_channels_layer2 = 64
W_conv2 = tf.Variable(weight_variable([
    filter_height, filter_width, out_channels_layer1, out_channels_layer2]))
b_conv2 = tf.Variable(bias_variable([out_channels_layer2]))

h_conv2 = tf.nn.relu(conv2D(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#########################  fully connected layer
out_fc1_layer = 1024
W_fc1 = tf.Variable(weight_variable([7*7*64,out_fc1_layer]))
b_fc1 = tf.Variable(bias_variable([out_fc1_layer]))

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#########################  dropout layer
keep_prob = tf.placeholder(dtype = tf.float32,name = 'keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#########################  out layer
out_fc2_layer = 10
W_fc2 = tf.Variable(weight_variable([out_fc1_layer,out_fc2_layer]))
b_fc2 = tf.Variable(bias_variable([out_fc2_layer]))

y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
# loss and optimizer
loss = -tf.reduce_sum(y_*tf.log(y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# accurancy esitmation
correct_pred = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accurancy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#test code
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(learning_epochs):
    for iteration in range(learning_iterations):
        images_feed, labels_feed = mnist.train.next_batch(batch_size)
        optimizer.run(feed_dict = 
            {x:images_feed,y_:labels_feed,keep_prob : 0.5})
        print("iteration:",iteration,"accurancy:")
    print("epoch:",epoch,
        "accurancy:",accurancy.eval(feed_dict = 
        {x:images_feed,y_:labels_feed,keep_prob : 1}))

# learning_epochs = 20
# batch_size = 200
# learning_iterations = int(traning_num / batch_size)