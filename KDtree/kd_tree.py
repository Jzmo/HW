import normal_2d
import numpy as np
import matplotlib.pyplot as plt

class KD_TREE():
    def __init__(self,data, leafsize = 10, isshow = False):
        self.data = data
        self.leafsize = int(leafsize)
        self.__show = isshow
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
            max = np.amax(data,axis = 0)
            min = np.amin(data,axis = 0)
            print('min:',min,'max:',max)
            split_dim = np.argmax(max - min)
            data = data[data[:,split_dim].argsort()]
            split_pos = int(data.shape[0] / 2)
            median = data[split_pos]
            less = data[:split_pos]
            more = data[split_pos+1:]
            if self.__show :
                print('split_dim:',split_dim,'median:',median)
                if split_dim == 0:
                    print('min1:',min[1])
                    plt.axvline(x = median[0],
                        ymin = min[1]/9, ymax = max[1]/9)
                else:
                    plt.axhline(y = median[1],
                        xmin = min[0]/9, xmax = max[0]/9)
            node = self.INNER_NODE(split_dim, median, 
                self.__built(less), self.__built(more))              
        return node

    
np.random.seed(12)
mu = np.array([[5, 5]])
Sigma = np.array([[3, 4], [1.5, 3]])
arr_train = normal_2d.generate(mu,Sigma,100)
arr_test = normal_2d.generate(mu,Sigma,5)

# print(arr_train)
# arr_train = arr_train[arr_train[:,0].argsort()]
# split_pos = arr_train.shape[0] // 2
# m = arr_train[split_pos]
# print(arr_train)
# print(m)
isshow = True 
K = KD_TREE(arr_train,isshow = isshow)

if isshow:
    plt.scatter(arr_train[:,0],arr_train[:,1],marker = 'o')
    plt.show()
