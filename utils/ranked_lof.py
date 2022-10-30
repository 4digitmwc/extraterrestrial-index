from sklearn.neighbors import LocalOutlierFactor
import numpy as np

def euclidean_square(A: np.ndarray, B: np.ndarray):
    return np.sum((A - B) ** 2)

def elastic_euclidean(lmbda1: float, lmbda2: float, lmbda3: float):
    def _euclidean(A, B):
        return np.sqrt(euclidean_square(A, B) * lmbda1) + np.sqrt(np.sum(A ** 2) * lmbda2) + np.sum(np.abs(A)) * lmbda3
    
    return _euclidean

def elastic_euclidean_square(lmbda1: float, lmbda2: float, lmbda3: float):
    def _euclidean(A, B):
        return euclidean_square(A, B) * lmbda1 + np.sum(A ** 2) * lmbda2 + np.sum(np.abs(A)) * lmbda3
    
    return _euclidean

