This project is from [Deep Learning in Udacity](https://cn.udacity.com/course/deep-learning--ud730) as a after class task.  
This task including some preprocessing and learning by sklearn
### data
notMNIST is a subset of English letters in different fonts from A to J. Dataset consists of small hand-cleaned part, about 19k instances, and large uncleaned dataset, 500k instances. Two parts have approximately 0.5% and 6.5% label error rate.  
some examples of letter "A":  
![image](http://yaroslavvb.com/upload/notMNIST/nmn.png)
data details can be found in [here](http://yaroslavvb.blogspot.hk/2011/09/notmnist-dataset.html)
### preprocess
process details can be found in [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)  
**problem 1**  
Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded.  
```python
def peek_display(filefolders):
    img_num = 0
    fig = plt.figure()
    for letter_dir in filefolders:
        im_dir = os.path.join(letter_dir,np.random.choice(os.listdir(letter_dir)))
        img = mpimg.imread(im_dir)
        img_num = img_num + 1
        fig.add_subplot(1,len(filefolders),img_num)
        implot = plt.imshow(img)
    return None
    
# peek_display(train_folders)
# peek_display(test_folders)
# plt.show()
```    
**problem 2**   
Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.   
**problem 3**     
Another check: we expect the data to be balanced across classes. Verify that.  
**problem 4**   
Convince yourself that the data is still good after shuffling!   
**problem 5**  
By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it. Measure how much overlap there is between training, validation and test samples.

Optional questions:

What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.  
**problem 6**   
Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.

Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.

Optional question: train an off-the-shelf model on all the data!
### train methods
### result
