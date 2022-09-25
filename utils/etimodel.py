import numpy as np

class ETIAssumptionModel:
    """this model was built under the assumption that 
    we have every score from every player in every map"""
    def __init__(self):
        pass

    def _transform(self, x: np.ndarray):
        return np.exp(np.mean(x, axis=1) * np.sqrt(x.shape[-1]))
    
    def transform(self, x: np.ndarray):
        return self._transform(x)

class ETIOldModel(ETIAssumptionModel):
    """this model was built under the ETIAssumptionModel 
    assumptions but there are some censored scores"""
    def __init__(self, p=1):
        self.p = p

    @staticmethod
    def power_keep_sign(x, p):
        return np.sign(x) * (np.abs(x) ** p)
    
    @staticmethod
    def _standardize_multivariate(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        return (X - mu) / sigma

    def transform(self, x: np.ndarray):
        imputed_x = self.power_keep_sign(x.T, self.p)
        for i in range(len(imputed_x)):
            imputed_x[i, np.where(np.isnan(imputed_x[i]))] = np.nanmin(imputed_x[i])
        
        return self._transform(imputed_x.T)

class ETIModel(ETIAssumptionModel):
    def __init__(self):
        super().__init__()
    
    def _impute_min(self, X: np.ndarray, train=False):
        imputed_x = X.T
        if train:
            self.min = np.nanmin(imputed_x, axis=1)
        for i in range(len(imputed_x)):
            imputed_x[i, np.where(np.isnan(imputed_x[i]))] = self.min[i]
        
        return imputed_x.T
    
    @staticmethod
    def _standardize_multivariate(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        return (X - mu) / sigma

    def _standardize(self, X: np.ndarray, train=False):
        if train:
            self.mu = np.nanmean(X.T, axis=1)
            self.std = np.nanstd(X.T, axis=1, ddof=1)

        return self._standardize_multivariate(X, self.mu, self.std)
    
    def _transform(self, X: np.ndarray):
        return np.exp(np.mean(X, axis=1) * np.sqrt(X.shape[-1]))

    def fit(self, X: np.ndarray):
        x_ = self._impute_min(X, True)
        x_ = self._standardize(x_, True)
        x_ = self._transform(x_)

        return x_

    def transform(self, X: np.ndarray):
        x_ = self._impute_min(X)
        x_ = self._standardize(x_)
        x_ = self._transform(x_)

        return x_
