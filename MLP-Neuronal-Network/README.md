# MLP - Neuronal Network

## Task Description
Train an MLP neural network with at least one hidden layer to identify the digit in an image using PyTorch.

You have the MNIST dataset, which contains tens of thousands of such image examples (input) along with the corresponding digit (desired output).

## Explanation of the Solved Task

### 1. Introduction
Using PyTorch and NumPy, I developed three neural networks to identify the digit in an image. 
I created three networks to observe the differences between using a specialized library for 
building neural networks versus manually implementing them, with or without an optimizer. 
Each neural network is accompanied by a `.txt` file that stores the following details: 
``learning rate``, ``epochs``, ``batch size``, and ``accuracy``. All neural networks have two hidden 
layers with **392** and **100** neurons, activation functions ``ReLU`` and ``Softmax`` for the final layer, 
and the ``Cross-Entropy`` Loss cost function.

### 2. Project Structure
- **File `first_neuronal_network.py`**: Neural network built using the PyTorch framework. Significant differences were observed when using a bias versus not, and between simple SGD and SGD with momentum.
  1. *Optimizer* with *momentum = 0.90*, *learning rate = 0.01*: accuracy of **97.81%**
  2. *Optimizer* with *momentum = 0.90* and *dampening = 25%*, *learning rate = 0.01*: accuracy of **97.74%**
  3. *Simple optimizer* with *bias = False*, *learning rate = 0.01*: accuracy of **95.86%**
  4. *Simple optimizer* with *bias = True*, *learning rate = 0.01*: accuracy of **86.20%**
  5. *Optimizer* with *momentum = 0.99*, *learning rate = 0.01*: accuracy of **10.30%**
  
- **File `neuronal_network.py`**: Neural network manually built using only the NumPy library, with *learning rate = 0.001*, *epochs = 35*, *batch size = 25*, achieving an accuracy of **77.82%**.
  
- **File `nn-optimizer.py`**: Neural network manually built using only the NumPy library but with the **Stochastic Gradient Descent** optimizer, *learning rate = 0.001*, *epochs = 35*, *batch size = 25*, achieving an accuracy of **81.40%**.

### 3. Methodology
1. **Data Collection**: Data was collected from the MNIST dataset.
2. **Data Preprocessing**: The data was split into training, validation, and test sets.
3. **Algorithm Implementation**: Two hidden layers with **392** and **100** neurons were used, with activation functions `ReLU` and `Softmax` for the final layer, and the `Cross-Entropy Loss` cost function.


### 4.1. Results  
The benefits of using the PyTorch framework, optimized for performance, were evident, as were the efficient use of the optimizer and the negative impact of manual implementation on neural network performance. Additionally, using the **`SGD`** optimizer for estimating the gradient at each step brought significant improvement. Combining it with **`momentum`**, which accelerates in the gradient's direction and reduces oscillation, allowed the neural network to learn even better.


