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