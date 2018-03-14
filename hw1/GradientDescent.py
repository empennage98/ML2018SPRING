import numpy as np

class GradientDescent:
    def __init__(self):
        self.w = None
        
    def fit(self, feature, target, reg=0, eta=.1, niter=30000, plot=False):
        row, col = feature.shape
        w = np.zeros((col + 1, 1))
        
        eta /= row
        sigma = np.linalg.norm(w)+eta

        X = np.c_[np.ones((row, 1)), feature]
        y = target.reshape(-1,1)

        for i in range(niter):
            grad = X.T.dot(X.dot(w)-y) + reg*w
            w -= eta / (sigma**0.5) * grad
            sigma += np.linalg.norm(w)
        
        #Ein = np.sqrt(np.mean(np.square(X.dot(w)-y)))

        #print(Ein)
        self.w = w
        
    def predict(self, feature):
        row, col = feature.shape
        w = self.w
        
        X = np.c_[np.ones((row, 1)), feature]
        print(w)
        print(X.dot(w))
        return X.dot(w)