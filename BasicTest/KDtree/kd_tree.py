import normal_2d
import numpy as np

class TREE_NODE():
    def __init__(median, split, right, left):
        self.median = median
        self.split = split
        self.right = right
        self.left = left

class KD_TREE():
    def __init__(root,data):
        
        
np.random.seed(12)
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
arr_train = normal_2d.generate(mu,Sigma,9)
arr_test = normal_2d.generate(mu,Sigma,5)

print(arr_train)
arr_train = arr_train[arr_train[:,0].argsort()]
split_pos = arr_train.shape[0] // 2
m = arr_train[split_pos]
print(arr_train)
print(m)