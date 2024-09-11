import numpy as np
from engine import *
from nn import init_radom_weights
from numpy.lib.stride_tricks import sliding_window_view

class ConvLayer:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    
    self.weights = Tensor(data=np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
    self.bias = Tensor(data=np.zeros((out_channels, 1, 1)))

  def forward(self, input):
    if self.padding > 0:
      padded_input = np.pad(input.data, 
                            ((0,0), (self.padding,self.padding), (self.padding,self.padding)), 
                            mode='constant')
      input = Tensor(data=padded_input)

    output = input.conv(self.weights, stride=self.stride)

    output = Tensor(data=output.data + self.bias.data)

    return output

  def zero_grad(self):
    self.weights.zero_grad()
    self.bias.zero_grad()

class MaxPool:
  def __init__(self, pool_size, stride):
    self.pool_size = pool_size
    self.stride = stride

  def forward(self, input):
    if not isinstance(input, Tensor):
        input = Tensor(data=input)
      
    return input.max_pool(self.pool_size, self.stride)