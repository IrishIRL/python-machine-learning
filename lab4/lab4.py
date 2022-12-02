#!/usr/bin/env python3
"""lab4"""
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from common import describe_data, test_env


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df):

    # Features can be excluded by adding column name to list
    drop_columns = ['country']
    df = df.drop(labels=drop_columns, axis=1)

    columns = df.columns

    # Though it will be better to use StandardScaler in this task
    scaler = StandardScaler()
    rescaled_dataset_standard = scaler.fit_transform(df)

    # standardisation
    # we need to create a new dataframe with the column labels and the rescaled values
    dataframe_standard = pd.DataFrame(
        data=rescaled_dataset_standard, columns=columns)

    # Return features data frame and dependent variable
    return df, dataframe_standard


# FUNCTIONS HERE
def wcss_plot(dataframe_standard):
    wcss = []
    # I do not see the reason for using 15 max_clusters, so left 11.
    max_clusters = 11

    for i in range(1, max_clusters):
        k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
        k_means.fit(dataframe_standard)
        wcss.append(k_means.inertia_)

    plt.plot(range(1, max_clusters), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('results/figure_1.png')


def plot_clusters(X, y, figure, file=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    plt.figure(figure)

    for cluster in range(0, len(set(y))):
        plt.scatter(X[y == cluster, 0], X[y == cluster, 1],
                    s=5, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    plt.title(figure)
    # Remove axes numbers because those are not relevant for visualisation
    plt.xticks([])
    plt.yticks([])

    if file:
        plt.savefig(file, papertype='a4')

    plt.show()


def t_sne(df, dataframe_standard):
    # After some additional visual tests got into conclusion there is no need
    # for more than 3 clusters.
    n_clusters = 3
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_kmeans = k_means.fit_predict(dataframe_standard)

    # Visualise with t-sne
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(
        dataframe_standard)

    # There are no clusters. Create fake array with one cluster
    plot_clusters(X_tsne, np.full(X_tsne.shape[0], 0),
                  't-SNE visualisation without clusters')

    plot_clusters(X_tsne, y_kmeans, 'k means clusters with TSNE')

    # Visualise with PCA - but PCA primary goal is not visualisation
    X_pca = PCA(n_components=2, random_state=0).fit_transform(
        dataframe_standard)
    plot_clusters(X_pca, np.full(X_pca.shape[0], 0),
                  'PCA visualisation without clusters')
    plot_clusters(X_pca, y_kmeans,
                  'PCA visualisation with clusters')

    # Add cluster to data frame as last column and plot with pairplot,
    # useful for verification purposes.
    df['cluster'] = y_kmeans
    sns.pairplot(df[list(df.columns)], hue='cluster')
    plt.show()
    return df


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    # Read dataset file to pandas data frame
    country_data = read_csv('data/Country-data.csv')

    # Save dataset description to file in results directory
    describe_data.print_overview(
        country_data, file='results/country_data_overview.txt')
    describe_data.print_categorical(
        country_data, file='results/country_categorical_features.txt')

    # Preprocess dataset if needed
    dataframe, df_standard = preprocess_data(country_data)

    # Find possible suitable number of clusters with help of elbow method.
    # WCSS plot shall be saved to results folder for review.
    wcss_plot(df_standard)

    # Visualise dataset with help of t-SNE dimensions reduction to 2 dimensions.
    # # See class 9 materials and examples for details
    dataframe = t_sne(dataframe, df_standard)

    # Select suitable clustering algorithm for your business problem and data set.
    # Find clusters.
    # Visualise dataset with found clusters with help of t-SNE dimensions reduction by adding different colour and
    # symbol to each cluster.

    # table of clusters showing mean values per cluster and per feature
    clusters_table = pd.pivot_table(dataframe, index=['cluster'])
    print(clusters_table)

    print('Done')
