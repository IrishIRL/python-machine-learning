#!/usr/bin/env python3
"""lab3"""
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from common import describe_data, test_env
from common import classification_metrics


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # Ugly was to get uncategorical_columns
    uncategorical_columns = []
    for column in (df):
        if not column in categorical_columns:
            uncategorical_columns.append(column)

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    for column in (categorical_columns):
        df = pd.get_dummies(df, prefix=[column], columns=[
                            column], drop_first=True)

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES
    for column in (df):
        if column in uncategorical_columns:
            df[column] = df[column].fillna(value=0)
        else:
            df[column] = df[column].fillna(value='Missing')

    df.drop(columns=df.columns[0], axis=1, inplace=True)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER
def logistic_regression(students_X, students_y):

    title = 'Logistic Regression'

    X_train, X_test, y_train, y_test = train_test_split(
        students_X, students_y, test_size=0.26, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # clf = LogisticRegression(random_state=0, solver='liblinear')
    # You should use other solvers than liblinear and multinomial multi_class.
    # use multinomial or not to use?? I suppose use other than multinomial, so not this:
    # clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial',
    #                         penalty='l2', max_iter=1000)
    # but this:
    clf = LogisticRegression(random_state=0, solver='saga', multi_class='ovr',
                             penalty='elasticnet', max_iter=1000, l1_ratio=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classification_metrics.print_metrics(y_test, y_pred, label=title)

    # print(title + ' accuracy:', accuracy_score(y_test, y_pred))


def knn(X, y):
    title = 'k-nn'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Try to increase number of neighbours, but not too much.
    clf = KNeighborsClassifier(n_neighbors=8, p=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classification_metrics.print_metrics(y_test, y_pred, label=title)

    # print(title + ' accuracy:', accuracy_score(y_test, y_pred))


def svm(X, y):
    title = 'SVC'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # clf = SVC(kernel='rbf', random_state=0)
    # Try to use sigmoid kernel with small gamma (the smaller I do it, the more distant
    # the result is), bigger tolerances (1e-3 default) and with probability estimates.
    # this is the closest I got to your results, but the accuracy dropped by that.
    clf = SVC(kernel='sigmoid', gamma=0.3, tol=0.01,
              C=0.1, probability=True, random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classification_metrics.print_metrics(y_test, y_pred, label=title)

    # print(title + ' accuracy:', accuracy_score(y_test, y_pred))


def naive_bayes(X, y):
    title = 'Naive Bayes'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Scaling is needed to plot
    # sc = StandardScaler()
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # clf = GaussianNB()
    # Instead GaussianNB use MultinomialNB or ComplementNB.
    # clf = MultinomialNB()
    clf = ComplementNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classification_metrics.print_metrics(y_test, y_pred, title)

    # print(title + ' accuracy:', accuracy_score(y_test, y_pred))


def decision_tree(X, y):
    title = 'Decision tree classifier'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    regressor = DecisionTreeClassifier(random_state=0)
    regressor.fit(X_train, y_train)
    classification_metrics.print_metrics(
        y_test, regressor.predict(X_test), title)

    # print(title + ' accuracy:', accuracy_score(y_test, regressor.predict(X_test)))


def random_forest(X, y):
    title = 'Random forest classifier'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Try with more than 10 estimators.
    regressor = RandomForestClassifier(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)

    classification_metrics.print_metrics(
        y_test, regressor.predict(X_test), title)

    # print(title + ' accuracy:', accuracy_score(y_test, regressor.predict(X_test)))


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')
    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT
    describe_data.print_overview(
        students, file='results/students_overview.txt')
    describe_data.print_categorical(
        students, file='results/students_categorical_features.txt')

    # Dropped students description
    drop_students = students[(
        students['In university after 4 semesters'] == 'No')]
    describe_data.print_overview(
        drop_students, file='results/drop_students_overview.txt')
    describe_data.print_categorical(
        drop_students, file='results/drop_students_categorical_features.txt')

    students_X, students_y = preprocess_data(students)

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    logistic_regression(students_X, students_y)
    knn(students_X, students_y)
    svm(students_X, students_y)
    naive_bayes(students_X, students_y)
    decision_tree(students_X, students_y)
    random_forest(students_X, students_y)

    print('Done')
