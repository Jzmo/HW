# download and extrict the traning and testing files
import input_data
mnist = input_data.read_data_sets('./',one_hot = True)

# import other package
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

img_size = 28
n_output = 10
n_feature = img_size*img_size
traning_num = 55000
validation_num = 5000
test_num = 10000

def vector_2_image(img_vector, img_height, img_width):
    total_length = img_vector.shape[0]
    if total_length != img_height * img_width:
        raise Exception(
            "image size doesn't agree with the height and width:",
            total_length,"with",img_height * img_width)
    img = np.zeros((img_height, img_width),dtype = np.float64)
    w = 0
    for h in range(img_height):
        img[h,:] = img_vector[h * img_width : 
            (h+1) * img_width]
            
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
learning_rate = 0.001
learning_epochs = 20
batch_size = 200
learning_iterations = int(traning_num / batch_size)

#set variables:
W = tf.Variable(np.zeros([img_size*img_size,n_output]))
b = tf.Variable(np.zeros(n_output,))

#set placeholder
x = tf.placeholder(dtype = np.float64,name = 'x',shape = [None,n_feature])
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(dtype = np.float64,name = 'y_',shape = [None,n_output])

loss = -tf.reduce_sum(y_*tf.log(y))

# set net choice
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#accurancy
right_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accurancy = tf.reduce_mean(tf.cast(right_pred,np.float64))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(learning_epochs):
        for iteration  in range(learning_iterations):
            images_feed, labels_feed = mnist.train.next_batch(batch_size)
            type(images_feed)
            sess.run(optimizer,feed_dict = {x:images_feed,y_:labels_feed})
        #log information
        print("epoch:",epoch,
            "accurancy:",sess.run(accurancy,feed_dict = {x:images_feed,y_:labels_feed}))

    