#!/usr/bin/env python3
"""lab2"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from common import feature_selection as feat_sel
from common import test_env

# Pylint error W0621: Redefining name 'X' from outer scope (line 61) (redefined-outer-name)
# pylint: disable=redefined-outer-name


def print_metrics(y_true, y_pred, label):
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R squared: %.2f' % (label, r2_score(y_true, y_pred)))


def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Linear regression all in
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(
        X_sel, y, print_text='Linear regression with feature selection')


# POLYNOMIAL REGRESSION
def polynomial_regression(X, y):
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    return linear_regression(
        X_poly, y, print_text='Polynomial regression')


# SVR
def svr(X, y):
    sc = StandardScaler()

    X = sc.fit_transform(X)
    y = sc.fit_transform(np.expand_dims(y, axis=1))

    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    reg = SVR(kernel='rbf', gamma='auto')
    reg.fit(X_train, y_train.ravel())

    print_metrics(np.squeeze(y_test), np.squeeze(reg.predict(X_test)), 'SVR')


# DECISION TREE REGRESSION
def tree_regression(X, y):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    reg = DecisionTreeRegressor(
        max_depth=None, min_samples_split=2, min_samples_leaf=1)
    reg.fit(X_train, y_train)

    print_metrics(y_test, reg.predict(X_test), 'Tree regression')


# RANDOM FOREST REGRESSION
def random_forest_regression(X, y):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    reg = RandomForestRegressor(n_estimators=10)
    reg.fit(X_train, y_train)

    print_metrics(y_test, reg.predict(X_test), 'Random forest regression')


if __name__ == '__main__':
    test_env.versions(['numpy', 'statsmodels', 'sklearn'])

    # https://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset
    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    polynomial_regression(X, y)
    svr(X, y)
    tree_regression(X, y)
    random_forest_regression(X, y)

    print('Done')
