# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = 'F:\\Jzmo\\tf\\1'

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