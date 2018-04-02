import numpy as np


class ConvLayer():
    def __init__(self, input, filter, stride):
        self.filter = filter
        self.filter_size = filter.shape[0]
        self.stride = stride
        self.data_width = input.shape[2]
        self.data_height = input.shape[1]
        self.data_depth = input.shape[0]
        self.input = input
        self.output_width = int((input.shape[2] - 1) / stride + 1)
        self.output_height = int((input.shape[1] - 1) / stride + 1)
        self.output = np.zeros((self.data_depth,
                                self.output_height,
                                self.output_width),
                                dtype = np.float64)
    
    def forward(self):
        zero_padding_size = int((self.filter_size-1)/2)
        self.zero_padding(zero_padding_size)
        for index in range(self.data_depth):
            self.output[index,:,:] = self.conv2D(self.input[index,:,:],self.filter,self.stride)
        return self.output
    
    def zero_padding(self,zero_padding_size):
        zero_pad = np.zeros((self.data_depth,
                            2*zero_padding_size+self.data_height,
                            2*zero_padding_size+self.data_width),
                            dtype = np.float64)
        print(self.input)                    
        zero_pad[:,
                 zero_padding_size:self.data_height+zero_padding_size,
                 zero_padding_size:self.data_width+zero_padding_size] = self.input
        self.input = zero_pad
        
    def conv2D(self,origin,filter,stride):
        output = np.zeros((self.output_height,self.output_width))
        for x in range(self.output_height):
            for y in range(self.output_width):
                output[x,y] = np.sum(np.multiply(filter,
                                   origin[x*stride:x*stride+self.filter_size,
                                          y*stride:y*stride+self.filter_size]))
        return output
        

class PoolingLayer():
    def __init__(self, input, method, pool_size):
        self.input = input
        self.method = method
        self.pool_size = pool_size
        self.data_width = input.shape[2]
        self.data_height = input.shape[1]
        self.data_depth = input.shape[0]
        self.out_data_width = self.data_width - self.pool_size+1
        self.out_data_height = self.data_height - self.pool_size+1
        self.output = np.zeros((self.data_depth,
                                self.out_data_width,
                                self.out_data_height))
    def MaxPooling(self,origin,output):
        for x in range(self.out_data_width):
            for y in range(self.out_data_height):
                output[x,y] = np.max(origin[
                                     x:x+self.pool_size,
                                     y:y+self.pool_size]) 
        return output
        
    def AveragePooling(self,origin,output):
        for x in range(self.out_data_width):
            for y in range(self.out_data_height):
                output[x,y] = np.average(origin[x:x+self.pool_size,
                                                y:y+self.pool_size])
        return output
        
    def Pooling2D(self,origin,output):
        if self.method == 'MaxPooling':
            self.MaxPooling(origin,output);
        elif self.method == 'AveragePooling':
            self.AveragePooling(origin,output)
        else:
            print('No such pooling method:',self.method)
        
    def forward(self):
        for index in range(self.data_depth):
            self.Pooling2D(self.input[index,:,:],self.output[index,:,:])
        return self.output

class CNNLayer():
    def __init__(self,input,filter,activitor,
                 pool_method,pool_size):
        self.input = input
        self.filter = filter
        self.activitor = activitor
        self.pool_method = pool_method
        self.pool_size = pool_size
        
def main_test():
    input = np.array(
        [[[1,0,1,0,2],
          [1,1,1,0,1],
          [1,0,1,0,0],
          [0,0,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    f1 = np.array([[1,-1,1],
                   [1,-1,-1],
                   [1,-1,-1]])
    c1 = ConvLayer(input,f1,1)
    p1 = PoolingLayer(c1.forward(),"MaxPooling",2) 
    print(p1.forward())  

if __name__ == "__main__":
    main_test()