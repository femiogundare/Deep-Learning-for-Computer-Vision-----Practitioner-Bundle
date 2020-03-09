import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        #initialize the weights and the learning rate
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha
    
    def step_function(self, x):
        #applying the step function where x>0 is 1, while contrary is 0
        return 1 if x>0 else 0
    
    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in range(epochs):
            for x, target in zip(X, y):
                pred = self.step_function(np.dot(x, self.W))
                
                #only perform weight update if prediction does not match target:
                if pred != target:
                    error = pred - target
                    
                    #update the weight matrix
                    self.W += -self.alpha*error*x
                
    def predict(self, X, add_bias=True):
        #ensure the input is a matrix
        X = np.atleast_2d(X)
        
        #check to see if the bias column needs to be added
        if add_bias:
            # Insert a column of 1's as the last entry in the feature matrix
            X = np.c_[X, np.ones(X.shape[0])]
        return self.step_function(np.dot(X, self.W))
            