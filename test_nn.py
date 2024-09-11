from engine import *
import numpy as np
from datasets import load_from_disk
from nn import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def softmax(x):
  x_max = np.max(x, axis=-1, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == '__main__':
  ds = load_from_disk('nums.hf')

  input_shape = 28*28
  hidden_layers = 2
  hidden_units = 120
  output_shape = 10
  nn = NeuralNet(input_shape, hidden_layers, hidden_units, output_shape)

  for i in tqdm(range(len(ds['train']))):
    x = np.array(ds['train'][i]['image']).flatten()
    y_true = np.zeros(output_shape)
    label = ds['train'][i]['label']
    y_true[ds['train'][i]['label']] = 1

    output = nn.forward(x)
    loss = softmax(output.data) - y_true.reshape(1, -1)
    output.backward(grad=loss)
    nn.update_weights(lr=0.001)

  y_true_list = []
  y_pred_list = []

  for i in tqdm(range(len(ds['test']))):
    x = np.array(ds['test'][i]['image']).flatten() 
    y_true = np.zeros(output_shape)
    label = ds['test'][i]['label']  
    y_true_list.append(label)
    y_true[ds['test'][i]['label']] = 1

    output = nn.forward(x)

    y_pred = np.argmax(softmax(output.data))
    y_pred_list.append(y_pred)

  print(accuracy_score(y_true_list, y_pred_list))