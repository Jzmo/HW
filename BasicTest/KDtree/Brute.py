import tensorflow as tf
import numpy as np
import input_data

image_size = 784
train_size = 5000
test_size = 500
right_pred = 0

mnist = input_data.read_data_sets('./',one_hot = True)
train_images, train_labels = mnist.train.next_batch(train_size)
test_images, test_labels = mnist.test.next_batch(test_size)


def brute_force(x_input, dataset, k ):
    distances = tf.reduce_sum(tf.pow(tf.subtract(dataset, x_input),2),1)
    k_value, k_index = tf.nn.top_k(-distances,k = k)
    
    return k_value, k_index
    


g = tf.Graph()


with g.as_default():

    with tf.name_scope('input'):
        x = tf.placeholder(dtype = 'float',name = 'x')
        dataset = tf.placeholder(dtype = 'float',name = 'dataset')
        y_ = tf.placeholder(dtype = 'float',name = 'y_') 
        k = tf.placeholder(dtype = 'int32',name = 'k')
        input_labels = tf.placeholder(dtype = 'float',name = 'input_labels')
    with tf.name_scope('method'):
        k_value, k_index = brute_force(x,dataset,k)
        k_labels = tf.placeholder(dtype = 'float',name = 'k_labels',shape = [10,])
        
    with tf.name_scope('output'):
        
        k_labels = tf.gather(input_labels,k_index)
        count_k_labels = tf.reduce_sum(k_labels,axis = 0)

        y = tf.argmax(count_k_labels)
        
        right = tf.placeholder('float',name = 'right')
        accurancy = tf.divide(right,test_size)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    for index in range(len(test_images)):
        
        label = np.argmax(test_labels[index])

        pred = sess.run(y,feed_dict = {
            input_labels:train_labels,
            x:test_images[index],
            dataset:train_images,
            k:1})
        print('test_num:',index,'label:',label,'prediction:',pred,' ',label == pred)
        if label == pred:
            right_pred += 1
    ac = sess.run(accurancy,feed_dict = {right:right_pred})
    print('accurancy:',ac)