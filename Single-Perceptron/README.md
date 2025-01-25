# Single Layer Perceptron

## Task Description
Train 10 perceptrons to identify the digit in an image.

Essentially, we will have one perceptron for each digit, which will learn to differentiate between its respective digit and the rest. By combining these perceptrons, we can deduce the digit present in a specific image.

You have the MNIST dataset, which contains tens of thousands of such image examples (input), along with the corresponding digit (the desired output of the system).

## Explanation of the Solved Task

### 1. Introduction
I used both the Simple Perceptron and ADALINE.

### 2. Project Structure
- File ``single-perceptron-layer.py``: Contains a Perceptron class implementing the Simple Perceptron algorithm.
- File ``adaline-single-perceptron-layer.py``: Adapts the code to use the features of the ADALINE perceptron.

### 3. Metodologie
1. **Data Collection**: Data was gathered from the MNIST dataset.
2. **Data Preprocessing**: The data was split into training, validation, and test sets.
3. **Algorithm Implementation**:
   - Implemented a ``Perceptron`` class where the learning rate, epochs, number of inputs, and number of perceptrons (10 for each digit from 0 to 9) were initialized.
   - This class contains:
     - A ``prediction`` function that uses a step or linear function.
     - An ``evaluate`` function to compute accuracy for each epoch.
     - A ``train`` function where the model is trained, errors are computed based on the perceptron type (Simple or ADALINE), and the bias and weight vector are updated.

### 4.1. Results - Simple Perceptron
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 0.8196              |
| 2     | 0.8283              |
| 3     | 0.8282              |
| 4     | 0.8184              |
| 5     | 0.8104              |
| 6     | 0.834               |
| 7     | 0.8117              |
| 8     | 0.8281              |
| 9     | 0.8364              |
| 10    | 0.7995              |

![image](https://github.com/user-attachments/assets/da23a087-9abb-4b4f-aa72-77003e15d541)


### 4.2. Results - ADALINE Perceptron
| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 0.7368              |
| 2     | 0.7370              |
| 3     | 0.7379              |
| 4     | 0.7378              |
| 5     | 0.7380              |
| 6     | 0.7378              |
| 7     | 0.7375              |
| 8     | 0.7378              |
| 9     | 0.7378              |
| 10    | 0.7381              |

![image](https://github.com/user-attachments/assets/8139132c-29da-49e8-9e8d-5e41c7645013)



