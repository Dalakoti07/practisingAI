# Fix the following points in head
# No of nodes/ neurons in input layer would be equal to no of features in dataset

import numpy as np 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def der_sigmoid(x):
    return x*(1-x)

class NeuralNetwork :
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backward(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def loss(self):
        return sum((self.y-self.output)*(self.y-self.output))

X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=np.array([[1],[1],[0]])

nn=NeuralNetwork(X,y)

# we are passing the data hundread times through NN
for i in range(100):
    nn.forward()
    nn.backward()
    print('error in '+str(i),end='')
    
    print(nn.loss())

print(nn.output)