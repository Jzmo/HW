# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import tarfile
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labeled examples. Given these sizes, it should be possible to train models quickly on any machine.

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = r"F:\Jzmo\tf\notMNIST"

def download_percent_hook(count, blockSize, totalSize):
    
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        
        last_percent_reported = percent
    
def maybe_download(filename, expected_bytes, force = False):
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print("Attempting to download:",filename)
        filename,_ = urlretrieve(url + filename, dest_filename, reporthook = download_percent_hook)
        print('\nDownload Complete')
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == expected_bytes:
            print("Found and verify" + dest_filename)
        else:
            raise Exception(
            "Failed to verify " + dest_filename + ". Can you get to it with a brower?"
            )
    return dest_filename
    
train_filename = maybe_download("notMNIST_large.tar.gz",247336696)
test_filename = maybe_download("notMNIST_small.tar.gz", 8458043)

num_classes = 10
np.random.seed(133)


# Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labeled A through J.
def maybe_extract(filename, force = False):
    #remove .tar .gz
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        print("%s already present - skinpping extraction of %s" % (root,filename))
    else:
        print("Extracting data for %s. This may take a while. Please wait." % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folder = [
        os.path.join(root,d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root,d))
        ]
    if len(data_folder) != num_classes:
        raise Exception(
        "Expected %d folders, one per class. Found %d instead" % (num_classes, len(data_folder)))
    print("data_folder")
    return data_folder
    
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)




# Problem 1
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.


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




# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.

# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.

# A few images might not be readable, we'll just skip them.

image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape =(len(image_files),image_size,image_size),
    dtype = np.float32)
    
    print(folder)
    num_image =0
    for image in image_files:
        imagefile = os.path.join(folder,image)
        try:
            image_data = (imageio.imread(imagefile).astype(float) -
                pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size,image_size):
                raise Exception("Unexpected image shape: %s" % str(image_data.shape))
            dataset[num_image,:,:] = image_data
            num_image = num_image + 1
        except(IOError,ValueError) as e:
            print("Could not read:",imagefile,':',e,'- it\'s ok, skipping.')           
    dataset = dataset[0:num_image,:,:]
    if num_image < min_num_images:
        raise Exception("Many fewer images than expected: %d <%d" % 
        num_image, min_num_images)
    
    print("Full dataset tensor: ",dataset.shape)
    print("Mean:",np.mean(dataset))
    print("Standard deviation:",np.std(dataset))
    return dataset
            
def maybe_pickle(data_folders, min_num_images_per_class, force = False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder+ '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print("%s alreadly pickled -- skipping" % set_filename)
        else:
            print("pickling %s..." % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename,'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unable to save data to ", set_filename,": ", e)
                
    return dataset_names
    
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# Problem 2
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.

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




# Problem 3
# Another check: we expect the data to be balanced across classes. Verify that.

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



# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune train_size as needed. The labels will be stored into a separate array of integers 0 through 9.

# Also create a validation dataset for hyperparameter tuning.

def make_array(nb_rows,img_size):

    if nb_rows:
        dataset = np.ndarray(shape = (nb_rows,img_size,img_size), dtype = np.float32)
        label = np.ndarray(shape = nb_rows, dtype = np.int32)
    else:
        dataset , label = None, None
    return dataset, label
    
    
def merge_datasets(pickle_files,train_size,valid_size = 0):
    
    num_classes = len(pickle_files)
    start_v , start_t = 0, 0
    end_v , end_t = valid_size, train_size
    
    valid_dataset, valid_label = make_array(num_classes*valid_size, image_size)
    train_dataset, train_label = make_array(num_classes*train_size, image_size)
    
    print(valid_dataset.shape)
    print(train_dataset.shape)
    
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file,'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_size is not None:
                    valid_set = letter_set[:valid_size,:,:]
                    valid_dataset[start_v:end_v,:,:] = valid_set
                    valid_label[start_v:end_v] = label
                    start_v += valid_size
                    end_v += valid_size
                    
                    train_letter = letter_set[valid_size:valid_size+train_size,:,:]
                    train_dataset[start_t:end_t] = train_letter
                    train_label[start_t:end_t] = label
                    start_t += train_size
                    end_t += train_size
                    
        except Exception as e:
            print('Unable to process data from ',pickle_file,':',e)
            raise
    return valid_dataset, valid_label, train_dataset, train_label

train_size = 100
valid_size = 10
test_size = 10

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets,train_size,valid_size)

_, _, test_dataset, test_labels = merge_datasets(
    test_datasets,test_size,test_size)
    
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation]
    labels = labels[permutation]
    return dataset, labels
    
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# Problem 4
# Convince yourself that the data is still good after shuffling!

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

# Finally, let's save the data for later reuse:

# pickled_file = os.path.join(data_root,'notMNIST.pickle')

# try:
    # f = open(pickled_file,'wb')
    # save = {
        # 'train_dataset':train_dataset,
        # 'train_labels': train_labels,
        # 'valid_dataset': valid_dataset,
        # 'valid_labels': valid_labels,
        # 'test_dataset': test_dataset,
        # 'test_labels': test_labels,
        # }
    # pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    # f.close()
# except Exception as e:
    # print('Unable to save data to', pickled_file)
    # raise

# statinfo = os.stat(pickled_file)
# print('Compressed pickle size:', statinfo.st_size)


# Problem 5
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it. Measure how much overlap there is between training, validation and test samples.

# Optional questions:

# What about near duplicates between datasets? (images that are almost identical)
# Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.


    
# Problem 6
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.

# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.

# Optional question: train an off-the-shelf model on all the data!

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