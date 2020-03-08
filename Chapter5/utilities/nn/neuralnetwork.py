import numpy as np


#Implementation of Back Propagation with an input layer, one hidden layer and an output layer
#consider the architecture to be 2-2-1


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        #initialize the weight matrix, network architecture and learning rate
        self.W = []
        self.layers = layers   #3 layers [2-2-1]
        self.alpha = alpha
        #OLUWAFEMI EMMANUEL OGUNDARE ----- 25/02/2020
        #start looping from the index of the first layer, but stop before reaching the last
        #2 layers
        for i in range(0, len(layers)-2):
            #randomly initialize the weight matrix connecting the nodes in each layer
            #add an extra node for the bias
            
            #each of the input node (3 nodes) in input layer must be connected to 3 other 
            #nodes in the hidden layer
            W = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(W / np.sqrt(layers[i]))   #normalize by sqrt of no of nodes in current layer which is 2
            
        #after completing the input layer, we are left with the hidden and output layers
        #the hidden layer requires a bias term while the output doesn't
        
        #each of the hidden nodes (3 nodes) in the hidden layer must be connected to just
        # the 1 node in the output layer
        W = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(W / np.sqrt(layers[-2]))  #normalize by sqrt of no of nodes in layer which is 2
        
    def __repr__(self):
        #function to return the architecture of the neural network
        return 'Neural Network {}'.format('-'.join(x) for x in self.layers)
    
    def sigmoid(self, x):
        # Compute the sigmoid activation value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # Compute the derivative of the sigmoid function assuming that 'x' has already been passed through the
        # sigmoid function
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000, display_update=100):
        #insert a column of 1's for the bias parameters
        X = np.c_[X, np.ones(X.shape[0])]
        
        for epoch in range(epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)  #function fit_partial is below
                
            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.compute_loss(X, y)
                print('[INFO]: epoch={}, loss={:.5f}'.format(epoch+1, loss))
    
    def fit_partial(self, x, y):
        #x is an individual data point; y is the corresponding label
        #take for instance, in our 2-2-1 architecture (3-3-1 when bias is added),
        #data points in the XOR gate are:
        # [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1] where the 1 in the last column is the
        #bias
        A = [np.atleast_2d(x)]  
        
        
        #A stores the activation functions on a normal day; but for a start, it contains only the chosen data point e.g [0, 0, 1]
        #for the input layer, the activation is that of the input itself;
        #for the hidden layer, the activation is appended updated and is the dot product of the input layer by the weight matrix
        #for the output layer, the activation is activated and is the dot product of the activation from the hidden layer with the weight matrix
        
        
        
        #FEED-FORWARD
        #loop over the layers of the network
        for layer in range(0, len(self.W)):   # which is from 0-2
            net = A[layer].dot(self.W[layer]) #for the first layer, it multiplies the data point by the weight matrix
            
            #applying the nonlinear sigmoid activation function to the output (which moves to the hidden layer):
            out = self.sigmoid(net)
            
            #add the activation to the list of activations
            A.append(out)
            
            
            
        #BACKPROPAGATION
        #the first step is to compute the difference between the 'prediction'(the final activation value)
        #and the actual target value
        error = A[-1] - y
        
        #apply chain rule and build a list of deltas
        #the first entry in the list of deltas is the error of the output layer multiplied by the
        #derivative of the activation function for the output
        D = [error*self.sigmoid_deriv(A[-1])]
        
        #loop over the layers in reverse order
        for layer in np.arange(len(A)-2, 0, -1):
            #the value of delta here will be the delta of the previous layer(output layer) dotted
            #with the weight of the current layer(hidden layer); this is followed by multiplying
            #with the derivative of the nonlinear activation function for the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])  #delta times derivative of activation function for the current layer
            D.append(delta)
            
        #reverse the order of the deltas since we looped in reverse order
        D = D[::-1]
            
            
        #WEIGHT UPDATE PHASE
        #loop over the layers
        for layer in np.arange(0, len(self.W)):     #which is from 0-2
                
            #update the weight by taking the product of the learning rate with the activation of the layer and its delta
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
                

    def predict(self, X, addBias=True):
        #converts the input to a matrix
        p = np.atleast_2d(X)
    
        #check to see if bias column should be added:
        if addBias:
            #insert columns of 1s
            p = np.c_[p, np.ones(p.shape[0])]
    
        #feed the matrix into the network by looping over the layers
        for layer in np.arange(0, len(self.W)):
            #compute the prediction
            p = self.sigmoid(np.dot(p, self.W[layer]))
        
        
        return p
    
    
    def compute_loss(self, X, targets):
        #convert the target to matrix
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # Return the loss
        return loss