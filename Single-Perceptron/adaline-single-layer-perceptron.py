import pickle, gzip, matplotlib.pyplot as plt, numpy as np

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

train_x, train_y = train_set
valid_x, valid_y = valid_set


class Perceptron:
    def __init__(self, learning_rate, epochs, number_of_inputs, number_of_perceptrons):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.number_of_inputs = number_of_inputs
        self.number_of_perceptrons = number_of_perceptrons
        self.weights = [np.zeros(number_of_inputs) for _ in range(number_of_perceptrons)]
        self.bias = np.zeros(number_of_perceptrons)


    def prediction(self, inputs, nr):
        net_input = np.dot(inputs, self.weights[nr]) + self.bias[nr]
        return net_input

    def evaluate(self, inputs, targets):
        correct_predictions = 0
        for i in range(len(inputs)):
            preactivations = [self.prediction(inputs[i], nr) for nr in range(self.number_of_perceptrons)]
            prediction_value = np.argmax(preactivations)
            if prediction_value == targets[i]:
                correct_predictions += 1
        return correct_predictions / len(inputs)

    def train(self, train_x, train_y, valid_x, valid_y):
        errors = np.zeros((self.number_of_perceptrons, self.epochs))
        for epoch in range(self.epochs):
            for nr in range(self.number_of_perceptrons):
                epoch_errors = []
                for i in range(len(train_x)):
                    inputs = train_x[i]
                    target = (train_y[i] == nr).astype(int)
                    prediction_value = self.prediction(inputs, nr)
                    error = target - prediction_value
                    epoch_errors.append(error)
                    self.weights[nr] += self.learning_rate * error * inputs
                    self.bias[nr] += self.learning_rate * error
                errors[nr, epoch] = np.mean(epoch_errors)
            validation_accuracy = self.evaluate(valid_x, valid_y)
            print(f'Epoch {epoch + 1}, Validation Accuracy: {validation_accuracy}')
        return errors


learning_rate = 0.01
epochs = 10
number_of_inputs = 784
number_of_perceptrons = 10

perceptron = Perceptron(learning_rate, epochs, number_of_inputs, number_of_perceptrons)
errors = perceptron.train(train_x, train_y, valid_x, valid_y)

# Plot the error for each perceptron
for i in range(number_of_perceptrons):
    plt.plot(errors[i], label=f'Perceptron {i}')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


