from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class LinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.model_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
