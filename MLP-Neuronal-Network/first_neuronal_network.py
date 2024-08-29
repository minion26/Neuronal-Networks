import pickle, gzip, torch

f = open("first_neuronal_network.txt", "a")
f.write("With momentum 0.99 \n")
f.write("With dampening 0.25 \n")

with gzip.open("../mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

train_x, train_y = torch.tensor(train_set[0], dtype=torch.float32), torch.tensor(train_set[1], dtype=torch.long)
valid_x, valid_y = torch.tensor(valid_set[0], dtype=torch.float32), torch.tensor(valid_set[1], dtype=torch.long)


class MyNN(torch.nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.fc1 = torch.nn.Linear(784, 392, bias=False)  # first hidden layer
        self.fc2 = torch.nn.Linear(392, 100, bias=False)  # second hidden layer
        self.fc3 = torch.nn.Linear(100, 10, bias=False)  # output layer
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')


    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        return x


def check_accuracy(inputs, labels):
    correct_predictions = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            outputs = model(batch_inputs)
            _, predictions = outputs.max(1)
            correct_predictions += (predictions == batch_labels).sum()
            num_samples += predictions.size(0)
        print(f'Got {correct_predictions}/{num_samples} with accuracy {float(correct_predictions) / float(num_samples) * 100: .2f}')
        f.write(f'Got {correct_predictions}/{num_samples} with accuracy {float(correct_predictions) / float(num_samples) * 100: .2f}\n')
    model.train()

model = MyNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.90, dampening=0.25)
f.write("learning_rate=0.01\n")
loss_function = torch.nn.CrossEntropyLoss()
batch_size = 20
f.write(f'batch_size={batch_size}\n')
inputs = train_x
labels = train_y
epoch = 35
for epoch in range(epoch):
    print(f'Epoch {epoch + 1}')
    f.write(f'Epoch {epoch + 1}\n')
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        outputs = model(batch_inputs)
        loss = loss_function(outputs, batch_labels)
        # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    check_accuracy(valid_x, valid_y)

f.write("\n\n")
f.write("\n\n")
f.close()


# check_accuracy(valid_x, valid_y)
