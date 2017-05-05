import numpy as np
import pdb


class Eszsl():

    def __init__(self,
                 S,
                 g,
                 l):
        self.S = S
        self.g = g
        self.l = l
        self.V = None

    def fit(self, X, y):
        """
        I follow sklearn's semantic: a row in X represents an instance
                                     with d dimensional features.
        The paper's semantic is the other way around.
        So it needs to be transposed at the beginning.
        y's semantic is same in the paper and sklearn.
        """
        X = np.transpose(X)
        # Add assertion stage here
        first = np.linalg.inv(X * np.transpose(X) \
                              + self.g * np.identity(X.shape[0]))
        second = X * y * np.transpose(self.S)
        third = np.linalg.inv(self.S * np.transpose(self.S) \
                              + self.l * np.identity(self.S.shape[0]))
        self.V = first * second * third
        self.VS = self.V * self.S

    def predict(self, X):
        predicted_Y = list()
        for x in X:
            pred_idx = np.argmax(x * self.VS)
            pred_y = [1 if i==pred_idx else 0 for i in range(0, self.S.shape[1])]
            predicted_Y.append(pred_y)
        return np.asarray(predicted_Y)
