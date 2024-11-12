import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Define the test function
def testMLP(net, x):
    n = len(x)
    x = np.concatenate((x,np.ones((n,1))),axis=1)
    s1 = sigmoid(np.matmul(x,net['w']))
    s1 = np.concatenate((s1,np.ones((n,1))),axis=1)
    s = np.matmul(s1,net['w_2'])
    return np.argmax(s,axis=1)

# Define the training process
def trainMLP(X_tr, Y_tr, M = 2, alpha = 0.1, epoch = 10000):
    '''
    Params:
        - X_tr: Training samples [#samples #features]
        - M: number of neurons in the hidden layer
        - alpha: learning rate
        - epoch: number of epochs
    Return:
        - net: a dict contains two keys {'w':'','w_2':''}
            - w weights for the input layer
            - w2 weights for the output layer
    '''
    # Initial parameters
    ntr = len(X_tr) # Number of training samples
    d = len(X_tr[0]) # Number of features
    # Initialization of the weigths for the input layer
    w = np.array(np.random.randn(d+1,M))
    # Initialization of the weigths for the output layer
    w_2 = np.array(np.random.randn(M+1, len(Y_tr[0])))
    # Add the bias
    X_trext_all = np.concatenate((X_tr,np.ones((ntr,1))),axis=1)
    Y_tr_all = Y_tr
    # Initialize the training loss and accuracy list
    training_loss = []
    training_acc = []
    # Training
    for i in range(epoch):
        index_list = list(range(X_trext_all.shape[0]))
        random.shuffle(index_list)
        for j in range(X_trext_all.shape[0]):
            
            X_trext, Y_train = X_trext_all[index_list[j],:], Y_tr_all[index_list[j],:]
            # Forward propagation
            z = sigmoid(np.matmul(X_trext,w))
            # z_ext = np.concatenate((z,np.ones((ntr,1))),axis=1)
            z_ext = np.concatenate((z,np.array([1])))
            y = np.matmul(z_ext, w_2)
            # Backpropagation
            delta_output = y - Y_train
            # print(y)
            # print(Y_train)
            grad_1 = sigmoidGradient(np.concatenate((np.matmul(X_trext,w),np.array([1]))))
            delta_input = np.matmul(w_2,delta_output.T)*grad_1.T
            delta_input = delta_input[:-1]
            grad_2 = np.matmul(z_ext.reshape((-1,1)),delta_output.reshape((1,-1)))
            w_2 = w_2 - alpha*grad_2
            grad = np.matmul(X_trext.reshape((-1,1)),delta_input.reshape((1,-1)))
            w = w - alpha*grad
        
        if i % 20 == 0:
            # Re-evaluation of the error
            Z_tr = sigmoid(np.matmul(X_trext_all,w))
            Z_trext = np.concatenate((Z_tr,np.ones((ntr,1))),axis=1)
            y_tr = np.matmul(Z_trext,w_2)
            train_acc = np.mean(np.equal(np.argmax(y_tr,axis=1), np.argmax(Y_tr_all,axis=1)))
            training_loss.append(np.sqrt(np.mean((y_tr-Y_tr_all)**2)))
            training_acc.append(train_acc)
            
    net = {'w':'','w_2':''}
    # Save the final weights
    net['w'] = w
    net['w_2'] = w_2
    return net, training_loss, training_acc

# Sigmoid function
def sigmoid(A):
    return 1/(1+np.exp(-A))

# Gradient of the sigmoid function
def sigmoidGradient(A):
    return sigmoid(A)*(1-sigmoid(A))

if __name__ == "__main__":
    # training input x
    x_tr = np.array([[0,0],[1,0],[0,1],[1,1]])
    # training label y (one hot)
    y_tr = np.array([[1,0],[0,1],[0,1],[1,0]])
    net,loss_lst,acc_lst = trainMLP(x_tr,y_tr)
    epoch_lst = list(np.arange(1,501))
    plt.plot(epoch_lst,loss_lst,'bo-')
    plt.legend(['training loss'])
    plt.show()
    plt.plot(epoch_lst,acc_lst,'go-')
    plt.legend(['training accuracy'])
    plt.show()
    # test data
    x_test = np.array([[1,1],[1,0],[0,0],[0,1]])
    y_test = np.array([[1,0],[0,1],[1,0],[0,1]])
    y_predict = testMLP(net, x_test)
    print('prediction of {} is {}'.format(x_test,y_predict))
    print('test accuracy is {}'.format(np.mean(np.equal(np.argmax(y_test,axis=1),y_predict))))