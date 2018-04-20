import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt 

def generate(mu,Sigma,sampleNo):

    R = cholesky(Sigma)
    arr = np.dot(np.random.randn(sampleNo,2),R)+mu

    return arr
    
def draw(mu,Sigma,sampleNo):    
    arr = generate(mu,Sigma,sampleNo)
    plt.plot(arr[:,0],arr[:,1],'bo')
    plt.show()
    
    return arr

def data(): 
    np.random.seed(12)
    mu = np.array([[5, 5]])
    Sigma = np.array([[3, 4], [1.5, 3]])
    train = generate(mu,Sigma,20)
    test = generate(mu,Sigma,5)
    
    return train, train_label, test, test_label