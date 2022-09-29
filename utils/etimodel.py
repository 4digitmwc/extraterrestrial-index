import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LogisticRegression

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
    _imputing_techniques = {
        'min': lambda x: np.nanmin(x, axis=1),
        'max': lambda x: np.nanmax(x, axis=1),
        'mean': lambda x: np.nanmean(x, axis=1),
        'median': lambda x: np.nanmedian(x, axis=1)
    }
    def __init__(self, imputing_technique='min'):
        super().__init__()
        if isinstance(imputing_technique, str):
            self._imputing_technique = self._imputing_techniques[imputing_technique]
        else:
            self._imputing_technique = imputing_technique
    
    def _impute_min(self, X: np.ndarray, train=False):
        imputed_x = X.T
        if train:
            self.min = self._imputing_technique(imputed_x)
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
        y = np.mean(X, axis=1) * np.sqrt(X.shape[-1])
        y = np.exp(y)
        return y

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

class GeneralizedETIModel:
    def __init__(self, _4dm_records: pd.DataFrame, beatmap_categories: list, lasso_alpha=1):
        self.beatmap_categories = beatmap_categories
        self.main_eti_models = {cat: ETIModel() for cat in beatmap_categories}
        self.tournament_eti_models = {cat: ETIModel() for cat in beatmap_categories}
        self._4dm_records = _4dm_records
        self._linearETIRegression = Lasso(lasso_alpha)
        self.fit_4dm()

    @staticmethod
    def filter_pivot(records: pd.DataFrame, category: str):
        filtered_records = records[records['beatmap_type'] == category]
        filtered_records['beatmap'] = records.apply(lambda x: x['round'] + "_" + x['beatmap_type'] + "_" + x['beatmap_tag'], axis=1)
        return filtered_records.pivot('player_name', 'beatmap', 'score_logit')
    
    def fit_eti(self, tournament_records: pd.DataFrame):
        for category in self.beatmap_categories:
            table = self.filter_pivot(tournament_records, category)
            self.tournament_eti_models[category].fit(table.values)
    
    def fit_4dm(self):
        for category in self.beatmap_categories:
            table = self.filter_pivot(self._4dm_records, category)
            self.main_eti_models[category].fit(table.values)
    
    def transform_eti(self, tournament_records: pd.DataFrame):
        eti_cats = pd.DataFrame(index=tournament_records['player_name'].unique(), columns=self.beatmap_categories)
        for category in self.beatmap_categories:
            table = self.filter_pivot(tournament_records, category)
            etis = self.tournament_eti_models[category].transform(table.values)
            eti_cats[category] = pd.DataFrame(etis, index=table.index)
            eti_cats[category].fillna(np.nanmin(etis))
        return eti_cats
    
    def _4dm_eti(self):
        eti_cats = pd.DataFrame(index=self._4dm_records['player_name'].unique(), columns=self.beatmap_categories)
        for category in self.beatmap_categories:
            table = self.filter_pivot(self._4dm_records, category)
            etis = self.main_eti_models[category].transform(table.values)
            eti_cats[category] = pd.DataFrame(etis, index=table.index)
            eti_cats[category].fillna(np.nanmin(etis))
        return eti_cats
    
    
    def fit_regression(self, tournament_records: pd.DataFrame):
        # I AM THE STORM THAT IS APPROACHING
        # PROVOKING BLACK CLOUD IN ISOLATION

        # Fit the model for tournament first
        self.fit_eti(tournament_records)
        # Get ETI of 4dm
        _4dm_eti_result = self._4dm_eti()
        # Get Tournament ETI
        eti_cats_tournament = self.transform_eti(tournament_records)
        # Find Average 4dm ETI
        avg_eti = _4dm_eti_result.apply(lambda x: np.mean(x), axis=1)
        
        # Get all list of 4dm players who participated in that tournament
        list_players = list(set(avg_eti.index).union(eti_cats_tournament.index))

        try:
            assert len(list_players) > 5
        except AssertionError:
            raise AssertionError("Not enough 4dm4 Sample size")
        
        # Obtaining X and y for training data
        X = eti_cats_tournament.loc[list_players].values
        y = avg_eti.loc[list_players].values.flatten()

        # Fit the linear regression w L1 model
        self._linearETIRegression.fit(X, y)
    
    def predict(self, players_records: pd.DataFrame):
        eti_cats_players = self.transform_eti(players_records)
        return self._linearETIRegression.predict(eti_cats_players.values)
