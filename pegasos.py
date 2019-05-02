import numpy as np
from sklearn import preprocessing
import random
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


class Pegasos(object):
    def __init__(self, n_iter=10, lambda1=1, projection=False, bias = False):
        self.n_iter = n_iter
        self._lambda = lambda1
        self.projection = projection
        self.labelEncoder = None
        self.bias = bias
        self.w = np.zeros((X.shape[1], 1))
        print("Done")
 
    def fit(self, X, Y):
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(Y)
        Y_labels = 2*self.labelEncoder.transform(Y) - 1
        self.w = np.zeros((X.shape[1]))
        
        for t in range(self.n_iter):
            i_t =  random.randint(0, X.shape[0] - 1)
            x = X[i_t]
            y = Y_labels[i_t]
            
            eta = 1.0/float(self._lambda * (t+1))

            if((y*( self.w.dot(x)))  < 1):
                self.w = (1 - eta*self._lambda)*self.w + eta*y*((x))
            else:
                self.w = (1 - eta*self._lambda)*self.w
            if(self.projection):
                self.w = (np.min((((1.0/ np.sqrt(self._lambda))/np.linalg.norm(self.w)), 1.0))) * self.w
            
    def predict(self, X):
        Ypred = (((X)@self.w))
        Y =  clf.labelEncoder.inverse_transform(((1 + (np.sign(Ypred)))/2).astype(int))
        return Y
    
    def test(self, X, Y):
        Ypred = (((X)@self.w))
        Y =  clf.labelEncoder.inverse_transform(((1 + (np.sign(Ypred)))/2).astype(int))
        
        accuracy = accuracy_score(Y, testY)
        f1 = f1_score(Y, testY)
        
        print("Accuracy is : " + str(accuracy))
        print("F1 score is : " + str(f1))


class Mercer_Pegasos(Pegasos):
    def __init__(self, n_iter=10, lambda1=1, projection=False, bias = False, kernel = None):
        super().__init__(n_iter, lambda1, projection, bias)
        if not (kernel):
            kernel = self.rbf
        self.kernel = kernel
        
    def fit(self, X, Y):
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(Y)
        Y_labels = 2*self.labelEncoder.transform(Y) - 1
        self.w = np.zeros((X.shape[1]))
        self.alpha = np.zeros((X.shape[0]))
        self.X = X
        
        for t in range(self.n_iter):
            i_t =  random.randint(0, X.shape[0] - 1)
            x = X[i_t]
            y = Y_labels[i_t]
            
            eta = 1.0/float(self._lambda * (t+1))
            
            error = 0
            for j in range(X.shape[0]):
                error += self.alpha[j] * y * (self.kernel(x, X[j]))
            if( (y*(1/self._lambda)* error) < 1):
                self.alpha[i_t]+=1
                
    def predict(self, X):
        Ypred = np.zeros((X.shape[0]))
        for i in range(X.shape[0]) :
            wTx = 0
            for j in range(self.X.shape[0]) :
                wTx += self.alpha[j] * self.kernel(X[i], self.X[j])
            Ypred[i] = np.sign(wTx)
        Ypred[Ypred > 0 ] = 1
        Ypred[Ypred <= 0 ] = 0
        return Ypred
    
    def test(self, X, Y):
        Ypred = np.zeros((X.shape[0]))
        for i in range(X.shape[0]) :
            wTx = 0
            for j in range(self.X.shape[0]) :
                wTx += self.alpha[j] * self.kernel(X[i], self.X[j])
            Ypred[i] = np.sign(wTx)
        Ypred[Ypred > 0 ] = 1
        Ypred[Ypred <= 0 ] = 0

        Y =  clf.labelEncoder.inverse_transform(Ypred.astype(int))
        
        accuracy = accuracy_score(Y, testY)
        f1 = f1_score(Y, testY)
        
        print("Accuracy is : " + str(accuracy))
        print("F1 score is : " + str(f1))
            
            
    def rbf(self, x1, x2, gamma = 0.5):
        return np.exp(-gamma* np.linalg.norm(x1-x2) * np.linalg.norm(x1-x2))



