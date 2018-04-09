## notMNIST
This project is from [Deep Learning in Udacity](https://cn.udacity.com/course/deep-learning--ud730) as a after class task,  including some preprocessing and learning by sklearn.  
(methods including:Ridge Regression、Lasso、Multi-task Lasso)  
### data
notMNIST is a subset of English letters in different fonts from A to J. Dataset consists of small hand-cleaned part, about 19k instances, and large uncleaned dataset, 500k instances. Two parts have approximately 0.5% and 6.5% label error rate.  
Some examples of letter "A":  
![image](http://yaroslavvb.com/upload/notMNIST/nmn.png)  
Data details can be found in [here](http://yaroslavvb.blogspot.hk/2011/09/notmnist-dataset.html)
### preprocess
Note: Here only problems listed in the class are given. Detials of main process can be found in [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)  : 
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
```python  
def peek_display_pickled(datasets):
    fig = plt.figure()
    img_num = 0
    for data_set_name in datasets:
        if os.path.exists(data_set_name):
            with open(data_set_name,'rb') as f:
                dataset = pickle.load(f)
                img = dataset[np.random.randint(low = 0, high = len(dataset)),:,:]
                img_num = img_num +1
                fig.add_subplot(1,len(datasets),img_num)
                plt.imshow(img)
    return None

# peek_display_pickled(train_datasets)
# peek_display_pickled(test_datasets)
# plt.show()     
```   
**problem 3**     
Another check: we expect the data to be balanced across classes. Verify that.  
```  python
def isbalance(datasets):
    class_size = []
    class_num = 0
    isbalance  = False
    for data_set_name in datasets:
        if os.path.exists(data_set_name):
            with open(data_set_name,'rb') as f:
                dataset = pickle.load(f)
                class_size.append(len(dataset[0,:,:]))
                class_num += 1
    print(class_size)
    if (np.max(class_size)-np.min(class_size)) > np.mean(class_size) * 0.1 : 
        print("dataset is not balance")
        isbalance = False
    else:
        print("dataset is balance")
        isbalance = True
    return isbalance
    
# b1 = isbalance(train_datasets)
# b2 = isbalance(test_datasets)

# print("train_datasets is balance: ", b1)
# print("test_datasets is balance: ", b2
```   
**problem 4**   
Convince yourself that the data is still good after shuffling!   
```  python
def peek_display_shuffled(dataset,labels):
    dataset_num = len(dataset)
    img_index = np.random.randint(0,dataset_num)
    img = dataset[img_index,:,:]
    label = labels[img_index]
    plt.imshow(img)
    plt.title(label)
    plt.show()
    return None

# peek_display_shuffled(train_dataset,train_labels)
# peek_display_shuffled(valid_dataset,valid_labels)
# peek_display_shuffled(test_dataset,test_labels)
```  
``` python
classifier = {'L1 logistic':LogisticRegression(penalty = 'l1'),
            'L2 logistic':LogisticRegression(penalty = 'l2'),
            'L2 logistic(Multinomial)':LogisticRegression(penalty = 'l2',solver='lbfgs',multi_class='multinomial'),
            }

n_classifier = len(classifier)

n_features = image_size * image_size

# prepare train data
n_train_samples, _, _= train_dataset.shape
train_data = np.reshape(train_dataset,(n_train_samples, n_features))

# prepare valid data
n_valid_samples, _, _= valid_dataset.shape
valid_data = np.reshape(valid_dataset,(n_valid_samples, n_features))

# prepare test data
n_test_samples, _, _= test_dataset.shape
test_data = np.reshape(test_dataset,(n_test_samples, n_features))
            
for index,(name,classifier) in enumerate(classifier.items()):
    classifier.fit(train_data,train_labels)
            
    rate = classifier.score(test_data, test_labels)
    print("classifier rate for %s : %f " % (name, rate))
```  
### train methods
**Ridge Regression**  
>Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares.  
[see Ridge](https://en.wikipedia.org/wiki/Ridge)  

The objective function to minimize is:

* ![equation](http://latex.codecogs.com/gif.latex?\min&space;_\omega&space;\left&space;\|&space;X\omega&space;-&space;y&space;\right&space;\|{_{2}}^{2}&space;&plus;&space;\left&space;\|&space;\alpha&space;\right&space;\|{_{2}}^{2})  

![equation](http://latex.codecogs.com/gif.latex?\alpha&space;>&space;0)  is a complexity parameter that controls the amount of shrinkage. Ridge in scikit-learn takes x, y in its **fit** method and stores the coefficients w of the linear model in its **coef_** member  
**Lasso**    
>The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero weights.  
[see Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))  

The objective function to minimize is:

* ![equation](http://latex.codecogs.com/gif.latex?\min&space;_\omega&space;\frac{1}{2n_{samples}}\left&space;\|&space;X\omega-y&space;\right&space;\|{_{2}}^{2}&plus;\alpha\left&space;\|&space;\omega&space;\right&space;\|_{1})

**Multi-task Lasso**  
### result
