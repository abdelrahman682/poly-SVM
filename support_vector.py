import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.system("cls")
class LinearSVM :
    def __init__(self, learning_rate=.001, iterations=1000,C = 1,dual = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C
        self.dual = dual

    def gradient_asccent(self, x, y):
        num_samples, num_features = x.shape
        self.alpha = np.zeros(num_samples)
        for _ in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * (x.dot(x.T))
            gradinet = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradinet

        self.alpha = np.clip(self.alpha, 0, self.C)

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.w = np.zeros(num_features)
        self.b = 0
        if self.dual :
            self.gradient_asccent(x, y)
            indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
            for i in indexes_SV:
                self.w += self.alpha[i] * y[i] * x[i]
            for i in indexes_SV:
                self.b += y[i] - np.dot(self.w.T, x[i])

            self.b /= len(indexes_SV)
        else:
            for _ in range(self.iterations):
                condition = y * (x.dot(self.w)+ self.b)
                idx_miss_classifed_points = np.where(condition < 1)[0]
                d_w = self.w - self.C * y[idx_miss_classifed_points].dot(x[idx_miss_classifed_points])
                self.w -= self.learning_rate * d_w
                d_b = -self.C * np.sum(y[idx_miss_classifed_points])
                self.b -= self.learning_rate * d_b   
    
    def predict(self, X):
        hyper_plane = X.dot(self.w) + self.b
        result = np.where(hyper_plane >=0, 1, -1)
        return result
    
    def descion_function(self,X):
        hyper_plane = X.dot(self.w) + self.b
        return hyper_plane
    
    def score(self,X,y):
        p = self.predict(X)
        return np.mean(p == y)


class PolySVC:
    def __init__(self, learning_rate=.001, iterations=1000,C = 1,dual = False,kernal ="linear",degree = 2):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C
        self.dual = dual
        self.kernal = kernal
        self.degree = degree
        self.X_train = None
        self.y_train = None
        
    def kernal_function(self,x1,x2):
        if self.kernal == "linear":
            return x1.dot(x2.T)
        elif self.kernal == "poly":
            return (x1.dot(x2.T) + 1) ** self.degree
        else:
            raise ValueError("only 'linear' and 'poly' kernals are supported")

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        kernal_matrix = self.kernal_function(X,X)
        for _ in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * kernal_matrix
            gradinet = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradinet

        self.alpha = np.clip(self.alpha, 0, self.C)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        if self.kernal != 'linear':
            self.dual = True
        if self.dual :
            self.gradient_asccent(X, y)
            indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
            if self.kernal == "linear":
                for i in indexes_SV:
                    self.w += self.alpha[i] * y[i] * X[i]
                for i in indexes_SV:
                    self.b += y[i] - np.dot(self.w.T, X[i])
                self.b /= len(indexes_SV)
            else:
                for i in indexes_SV:
                    kernal_matrix = self.kernal_function(X,X[i])
                    self.b += y[i] - np.sum(self.alpha * y * kernal_matrix)
                self.b /= len(indexes_SV)
        else:
            for _ in range(self.iterations):
                condition = y * (X.dot(self.w)+ self.b)
                idx_miss_classifed_points = np.where(condition < 1)[0]
                d_w = self.w - self.C * y[idx_miss_classifed_points].dot(X[idx_miss_classifed_points])
                self.w -= self.learning_rate * d_w
                d_b = -self.C * np.sum(y[idx_miss_classifed_points])
                self.b -= self.learning_rate * d_b   
    
    def predict(self, X_new):
        hyper_plane = self.descion_function(X_new)
        result = np.where(hyper_plane >=0, 1, -1)
        return result
    
    def descion_function(self,X_new):
        kernal_matrix = self.kernal_function(X_new,self.X_train)
        hyper_plane = kernal_matrix.dot(self.alpha * self.y_train) + self.b
        return hyper_plane
    
    def score(self,X,y):
        p = self.predict(X)
        return np.mean(p == y)

