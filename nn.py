import numpy as np
from random import uniform
from engine import *

def init_radom_weights(shape1, shape2):
  bound = np.sqrt(6 / (shape1 + shape2))
  return Tensor(np.random.uniform(-bound, bound, (shape1, shape2)))

class NeuralNet(object):
    def __init__(self, input_shape, hidden_layers, hidden_units, output_shape):
      self.input_layer = init_radom_weights(input_shape, hidden_units)

      self.hidden_layers_dict = {}
      for i in range(hidden_layers):
        self.hidden_layers_dict[f'hidden_layer_{i}'] = init_radom_weights(hidden_units, hidden_units)

      self.output_layer = init_radom_weights(hidden_units, output_shape)

      self.bias_vec = [Tensor(np.zeros((1, hidden_units))) for _ in range(hidden_layers)]
      self.bias_vec.append(Tensor(np.zeros((1, output_shape))))

    def forward(self, x):
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

    def print_weights(self):
      print("Input Layer Weights:")
      print(self.input_layer)

      for i in range(len(self.hidden_layers_dict)):
        print(f"Hidden Layer {i + 1} Weights:")
        print(self.hidden_layers_dict[f'hidden_layer_{i}'])

      print("Output Layer Weights:")
      print(self.output_layer)