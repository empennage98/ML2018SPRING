import numpy as np

class LogisticRegression:
    def __init__(self, C=0, eta=1, gamma=0.9, n_iter=100):
        self.C_ = C
        self.gamma_ = gamma
        self.eta_ = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        n, d = X.shape

        w = np.zeros((d + 1, 1))
        g = 1e-8
        X_ = np.c_[np.ones((n, 1)), X]
        y_ = y.reshape(-1, 1)

        for i in range(self.n_iter):
            grad = (-X_.T.dot(y_ - 1 / (1 + np.exp(-X_.dot(w)))) + self.C_*w) / n
            g = self.gamma_ * g + (1 - self.gamma_) * np.linalg.norm(grad)
            w -= self.eta_ / (g ** 0.5) * grad

        self.w_ = w

        return self

    def predict(self, X):
        n, d = X.shape

        X_ = np.c_[np.ones((n, 1)), X]
        y_ = 1 / (1 + np.exp(-X_.dot(self.w_)))

        return y_.ravel()

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape

        X0, y0 = X[y == 0], y[y == 0]
        X1, y1 = X[y == 1], y[y == 1]

        mu0 = np.mean(X0, axis = 0)
        mu1 = np.mean(X1, axis = 0)

        cov0 = (X0 - mu0).T.dot(X0 - mu0)
        cov1 = (X1 - mu1).T.dot(X1 - mu1)
        cov  = (cov0 + cov1) / n

        self.ratio_  = np.array([1 - np.sum(y) / n, np.sum(y) / n])
        self.mu_     = np.array([mu0, mu1])
        self.covInv_ = np.linalg.pinv(cov)
        self.covDet_ = np.linalg.det(cov)

        return self

    def predict(self, X):
        n, d = X.shape

        # Don't calculate the doniminator because they are the same
        L0 = np.exp(-((X-self.mu_[0]).dot(self.covInv_) * (X-self.mu_[0])).sum(-1)) \
                * self.ratio_[0]
        L1 = np.exp(-((X-self.mu_[1]).dot(self.covInv_) * (X-self.mu_[1])).sum(-1)) \
                * self.ratio_[1]

        y = (L1 > L0).astype(int)

        return y
