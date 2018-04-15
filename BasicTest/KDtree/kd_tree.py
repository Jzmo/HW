import normal_2d
import numpy as np
import matplotlib.pyplot as plt

class KD_TREE():
    def __init__(self,data, leafsize = 10):
        self.data = data
        self.leafsize = int(leafsize)
        self.tree = self.__built()
    
    class INNER_NODE():
        def __init__(self, split_dim, median, less, more):
            self.median = median
            self.split_dim = split_dim
            self.left = less
            self.right = more
            self.children = less.children + more.children

    class LEAF_NODE():
        def __init__(self, data):
            self.median = data
            self.children = len(data)
    
    def __built(self, data =[]):
        if data == []:
            data = self.data
        if len(data) < self.leafsize:
            node = self.LEAF_NODE(data)
        else:
            split_dim = np.argmax(
                np.amax(data,axis = 0)-
                np.amin(data,axis = 0))
            data = data[data[:,split_dim].argsort()]
            split_pos = data.shape[0] // 2
            median = data[split_pos]
            less = data[:split_pos]
            more = data[split_pos+1,:]
            node = self.INNER_NODE(split_dim, median, 
                self.__built(less), self.__built(more))  
            
        return node
        
        
    
np.random.seed(12)
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
arr_train = normal_2d.generate(mu,Sigma,150)
arr_test = normal_2d.generate(mu,Sigma,5)

# print(arr_train)
# arr_train = arr_train[arr_train[:,0].argsort()]
# split_pos = arr_train.shape[0] // 2
# m = arr_train[split_pos]
# print(arr_train)
# print(m)
K = KD_TREE(arr_train)

plt.scatter(arr_train[:,0],arr_train[:,1],marker = 'o')

plt.show()
