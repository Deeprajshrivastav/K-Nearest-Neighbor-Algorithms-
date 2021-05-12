import numpy as np
from collections import Counter


def accuracy_score(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict) / len(y_true)
    return accuracy


def euclidean_distance(p, q):
    distance = np.sqrt(sum(q - p)**2)
    return distance

  
class kNearestNeighbor:
    def __init__(self, k):
        self.k = k

        
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y


    def predict(self, X):
        predicted_outputs = [self.predictClass(x) for x in X]
        return predicted_outputs


    def predictClass(self, x):
        distance = [euclidean_distance(x, x1) for x1 in self.X_train]
        nearestNeighbourIndices = np.argsort(distance)[:self.k]
        nearestNeighbourLabels = [self.Y_train[i] for i in nearestNeighbourIndices]
        predictedLabels = (Counter(nearestNeighbourLabels).most_common())[0][0]
        return predictedLabels
      
      
