import torch
import torchvision
from torch import nn, optim
from torchvision import transforms,datasets
import numpy as np
import matplotlib.pyplot as plt

print("2.Linear Regression and Softmax Classification")

print("2.1 Linear Regression")
print("(1) Training data is a sine function with uniformly distributed noise from -0.5 to 0.5, and the code for generating training data is as follows:")
print("Draw a scatter plot of the training data.")
n = 100
x_train = np.linspace(-3, 3, n)
y_train = np.sin(x_train) + np.random.uniform(-0.5, 0.5, n)
plt.plot(x_train,y_train,'ro')
plt.show()

print("(2) Use deep learning framework to implement linear regression model.")
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        bias = torch.tensor([1]).cuda()
        x = x.unsqueeze(0)
        x = torch.stack((x,bias),dim=1)
        x = x.reshape((1,2))
        output = self.fc1(x)
        return output

model = LinearModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

for epoch in range(40):
    optimizer.zero_grad()
    for x,y in zip(x_train,y_train):
        x, y = torch.tensor(x).float().cuda(), torch.tensor(y).float().cuda()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(loss.item())
    

print("(3) Draw the regression curve and the scatter plot of the training data. Compare the regression curve and scatter plot.")
y_hat = []
for x in x_train:
    x = torch.tensor(x).float().cuda()
    y_hat.append(model(x).detach().cpu().item())

plt.plot(x_train,y_train,'ro',x_train,y_hat)
plt.show()

print("2.2 Linear Regression (transforming raw data using polynomial function)")
print("(1) Training data as above")
print("(2) Use deep learning framework to implement linear regression model.")
class PolynomialLinearModel(nn.Module):
    def __init__(self):
        super(PolynomialLinearModel, self).__init__()
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        bias = torch.tensor([1.0]).cuda()
        x = x.unsqueeze(0)
        x = torch.stack((x,x**2,x**3,bias),dim=1)
        x = x.reshape((1,4))
        output = self.fc1(x)
        return output

model = PolynomialLinearModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

for epoch in range(1000):
    optimizer.zero_grad()
    for x,y in zip(x_train,y_train):
        x = torch.tensor(x).float().cuda()
        y = torch.tensor(y).float().cuda()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(loss.item())

print("(3) Draw the regression curve and the scatter plot of the training data. Compare the regression curve and scatter plot.")
y_hat = []
for x in x_train:
    x = torch.tensor(x).float().cuda()
    y_hat.append(model(x).detach().cpu().item())

plt.plot(x_train,y_train,'ro',x_train,y_hat)
plt.show()

print("2.3 Softmax Classification")
print("(1) Using the MNIST dataset, the pixels of each picture in the MNIST dataset are 28*28.")
print("(2) Model")
print("where the dimension of w is 784 × 10, and the dimension of b is 10.")
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.reshape((-1,28*28))
        output = self.fc1(x)
        return output

transform_list = transforms.Compose([transforms.Resize((28,28)),
                                transforms.ToTensor()])
dataset_train = datasets.MNIST('dataset/', 
                    train=True, 
                    download=True,
                    transform=transform_list)
dataset_test = datasets.MNIST('dataset/', 
            train=False, 
            download=True,
            transform=transform_list)
train_loader =  torch.utils.data.DataLoader(dataset=dataset_train,
                                            batch_size=64, 
                                            shuffle=True)
test_loader =  torch.utils.data.DataLoader(dataset=dataset_test,
                                            batch_size=64, 
                                            shuffle=True)

model = SoftmaxModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

loss_lst = []
acc_lst = []
length = len(dataset_train.data)
for epoch in range(10):
    optimizer.zero_grad()
    loss_epoch = 0
    correct_cnt = 0
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.cuda(), y_train.cuda()
        output = model(x_train)
        correct_cnt += torch.sum(torch.eq(torch.argmax(output, dim=1),y_train).float()).item()
        loss = criterion(output, y_train)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    loss_lst.append(loss_epoch)
    acc_lst.append(correct_cnt/length)
    print(loss_epoch)
    print(correct_cnt/length)
epoch_lst = list(np.arange(1,11))
plt.plot(epoch_lst,loss_lst,'bo-')
plt.legend(['training loss'])
plt.show()
plt.plot(epoch_lst,acc_lst,'go-')
plt.legend(['training accuracy'])
plt.show()
print("(3) Plot the accuracy and loss of the training process, and submit final test accuracy.")
length = len(dataset_test.data)
correct_cnt = 0
for x_test, y_test in test_loader:
    x_test, y_test = x_test.cuda(), y_test.cuda()
    output = model(x_test)
    correct_cnt += torch.sum(torch.eq(torch.argmax(output, dim=1),y_test).float()).item()
    loss = criterion(output, y_test)
    loss.backward()
    optimizer.step()

test_acc = correct_cnt/length
print('test accuracy : {:.2f}%'.format(test_acc*100))
