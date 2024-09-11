import numpy as np
from engine import *
from conv import *

def init_radom_weights(shape1, shape2):
  bound = np.sqrt(6 / (shape1 + shape2))
  return Tensor(np.random.uniform(-bound, bound, (shape1, shape2)))

class LeNet():
  def __init__(self, input_shape, hidden_units, hidden_layers, output_shape):
    self.conv_layer_1 = ConvLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
    self.max_pool_1 = MaxPool(pool_size=2, stride=2)

    self.conv_layer_2 = ConvLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.max_pool_2 = MaxPool(pool_size=2, stride=2)

    self.input_layer = init_radom_weights(input_shape, hidden_units)

    self.hidden_layers_dict = {}
    for i in range(hidden_layers):
      self.hidden_layers_dict[f'hidden_layer_{i}'] = init_radom_weights(hidden_units, hidden_units)

    self.output_layer = init_radom_weights(hidden_units, output_shape)

    self.bias_vec = [Tensor(np.zeros((1, hidden_units))) for _ in range(hidden_layers)]
    self.bias_vec.append(Tensor(np.zeros((1, output_shape))))

  def forward(self, x):

    # conv layers
    out_c1 = self.conv_layer_1.forward(x)
    out_c1_relu = out_c1.relu()
    out_mp1 = self.max_pool_1.forward(out_c1_relu)
    out_c2 = self.conv_layer_2.forward(out_mp1)
    out_c2_relu = out_c2.relu()
    out_mp2 = self.max_pool_2.forward(out_c2_relu)

    # nn layers
    x = out_mp2.tflatten()
    inter = []
    inter.append(x)

    x = x @ self.input_layer
    inter.append(x)

    x = x.relu()
    inter.append(x)

    for i in range(len(self.hidden_layers_dict)):
      x = x @ self.hidden_layers_dict[f'hidden_layer_{i}'] + self.bias_vec[i]
      inter.append(x)
      x = x.relu()
      inter.append(x)

    x = x @ self.output_layer + self.bias_vec[-1]
    inter.append(x)

    return x    
  
  def update_weights(self, lr):
    self.conv_layer_1.weights.data -= lr * self.conv_layer_1.weights.grad 
    self.conv_layer_1.weights.zero_grad()

    self.conv_layer_2.weights.data -= lr * self.conv_layer_2.weights.grad 
    self.conv_layer_2.weights.zero_grad()

    self.input_layer.data -= lr * self.input_layer.grad
    self.input_layer.zero_grad()

    self.output_layer.data -= lr * self.output_layer.grad
    self.output_layer.zero_grad()

    self.bias_vec[-1].data -= lr * self.bias_vec[-1].grad
    self.bias_vec[-1].zero_grad()

    for i in range(len(self.hidden_layers_dict)):
      self.hidden_layers_dict[f'hidden_layer_{i}'].data -= lr * self.hidden_layers_dict[f'hidden_layer_{i}'].grad
      self.hidden_layers_dict[f'hidden_layer_{i}'].zero_grad()

      self.bias_vec[i].data -= lr * self.bias_vec[i].grad
      self.bias_vec[i].zero_grad()
