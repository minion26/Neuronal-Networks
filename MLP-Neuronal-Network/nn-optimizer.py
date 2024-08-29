import pickle, gzip, matplotlib.pyplot as plt, numpy as np

f = open("nn-optimizer.txt", "a")


with gzip.open("../mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

train_x, train_y = train_set
valid_x, valid_y = valid_set

train_x = train_x / 255.0
valid_x = valid_x / 255.0

# learning_rate = 0.001



def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


train_y_encoded = one_hot_encode(train_y, 10)
train_y_encoded = train_y_encoded.astype(int)


def kaiming_normal(shape, fan_in):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def initialize_params(input_layer, hidden_layer1, hidden_layer2, output_layer):
    w1 = kaiming_normal((hidden_layer1, input_layer), input_layer)
    b1 = np.zeros((hidden_layer1,))
    w2 = kaiming_normal((hidden_layer2, hidden_layer1), hidden_layer1)
    b2 = np.zeros((hidden_layer2,))
    w3 = kaiming_normal((output_layer, hidden_layer2), hidden_layer2)
    b3 = np.zeros((output_layer,))
    return w1, b1, w2, b2, w3, b3

def sgd_optimizer(w, b, dw, db, lr):
    w -= lr * dw
    b -= lr * db
    return w, b


# functiile de activare
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / (e_x.sum(axis=1, keepdims=True) + 1e-8)


# functii derivate
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# def softmax_derivative(x):
#     return x * (1 - x)


def forward_propagation(inputs, w1, b1, w2, b2, w3, b3):
    dot_product = np.dot(inputs, w1.T) + b1
    hidden_layer1 = relu(dot_product)
    # print(hidden_layer1.shape, hidden_layer1)

    dot_product = np.dot(hidden_layer1, w2.T) + b2
    hidden_layer2 = relu(dot_product)
    # print(hidden_layer2.shape, hidden_layer2)

    dot_product = np.dot(hidden_layer2, w3.T) + b3
    output = softmax(dot_product)
    # print(output.shape, output)
    return hidden_layer1, hidden_layer2, output


def loss_function(output, target):
    output = np.clip(output, 1e-12, 1.0)
    loss = -np.sum(target * np.log(output)) / output.shape[0]
    return loss


def loss_function_derivative(output, target):
    return output - target


def backward_propagation(layers, hidden_layer1, hidden_layer2, output, batch_x, batch_y, learning_rate):
    # Calcularea erorilor pentru fiecare strat
    output_error = loss_function_derivative(output, batch_y)
    output_delta = output_error

    hidden_layer2_delta = np.dot(output_delta, layers['w3']) * relu_derivative(hidden_layer2)
    hidden_layer1_delta = np.dot(hidden_layer2_delta, layers['w2']) * relu_derivative(hidden_layer1)

    # Calcularea gradientilor
    dw3 = np.dot(output_delta.T, hidden_layer2)
    db3 = np.sum(output_delta, axis=0)

    dw2 = np.dot(hidden_layer2_delta.T, hidden_layer1)
    db2 = np.sum(hidden_layer2_delta, axis=0)

    dw1 = np.dot(hidden_layer1_delta.T, batch_x)
    db1 = np.sum(hidden_layer1_delta, axis=0)

    # Actualizarea greutăților și bias-urilor folosind SGD
    layers['w3'], layers['b3'] = sgd_optimizer(layers['w3'], layers['b3'], dw3, db3, learning_rate)
    layers['w2'], layers['b2'] = sgd_optimizer(layers['w2'], layers['b2'], dw2, db2, learning_rate)
    layers['w1'], layers['b1'] = sgd_optimizer(layers['w1'], layers['b1'], dw1, db1, learning_rate)

    return layers


def create_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


def check_accuracy(inputs, labels, model, w1, b1, w2, b2, w3, b3, batch_size):
    correct_predictions = 0
    num_samples = 0
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        _, _, outputs = model(batch_inputs, w1, b1, w2, b2, w3, b3)
        predictions = np.argmax(outputs, axis=1)
        correct_predictions += np.sum(predictions == batch_labels)
        num_samples += len(batch_inputs)
    accuracy = float(correct_predictions) / float(num_samples) * 100
    print(f'Got {correct_predictions}/{num_samples} with accuracy {accuracy: .2f}')
    f.write(f'Got {correct_predictions}/{num_samples} with accuracy {accuracy: .2f}\n')
    return accuracy


def train(train_x, train_y, num_epochs=35, batch_size=25, learning_rate=0.001):
    layers = {
        'w1': kaiming_normal((392, 784), 784),
        'b1': np.zeros((392,)),
        'w2': kaiming_normal((100, 392), 392),
        'b2': np.zeros((100,)),
        'w3': kaiming_normal((10, 100), 100),
        'b3': np.zeros((10,))
    }

    f.write("num_epochs=35\n")
    f.write("batch_size=25\n")
    f.write("learning_rate=0.001\n")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in create_batches(train_x, train_y_encoded, batch_size):
            a1, a2, output = forward_propagation(batch_x, layers['w1'], layers['b1'], layers['w2'], layers['b2'], layers['w3'], layers['b3'])
            epoch_loss += loss_function(output, batch_y)
            layers = backward_propagation(layers, a1, a2, output, batch_x, batch_y, learning_rate)
        epoch_loss /= len(train_x) / batch_size
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        f.write(f'Epoch {epoch + 1}, Loss: {epoch_loss}\n')

        # Evaluare pe setul de validare
        check_accuracy(valid_x, valid_y, forward_propagation, batch_size=batch_size, w1=layers['w1'], b1=layers['b1'], w2=layers['w2'], b2=layers['b2'], w3=layers['w3'], b3=layers['b3'])




train(train_x, train_y)
f.write("\n\n")
f.write("\n\n")
f.close()

