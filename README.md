# python-machine-learning
Machine Learning Labs

## LAB 1 - Development environment setup and dataset statistics
### The purpose of this assignment
The goal of this lab is to check that the student has knowledge in the following topics:
* Development environment setup.
* Working with datasets based on pandas and csv.
* Statistics refreshment.
* Dataset statistics and visualisation with pandas, numpy and matplotlib.

### Task
1. Read dataset in project template data directory to pandas data frame and print out data overview.
2. Print out all possible State column values and counts.
3. Print out all Type of Breach column unique values.
4. Create new pandas data frame and extract from previous data frame only Texas state breaches where type of breach contains Hacking/IT Incident.
5. Extract column (pandas series) Individuals Affected from previously created data frame.
6. Calculate number of individuals affected per breach mean, median, mode, standard deviation and quartiles and print out the results.
7. Plot Individuals affected per breach histogram including lines showing mean, median and mode.
8. Plot Individuals affected per breach box plot.

## LAB 2 - Regression models trials and metrics
### The purpose of this assignment
The goal of this lab is to check that the student has knowledge in the following topics:
* Multiple linear regression.
* Polynomial regression.
* Support vector regression.
* Decision tree regression.
* Random forest regression.
Example dataset [Boston house prices toy dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html) is used in this assignment. Dataset does not require preprocessing and it has no categorical features. Downside of using example dataset is that different regression algorithms metrics do not differ remarkably.

### Task
1. Create matrix of features(X) and dependent variable(y).
2. Split dataset into training and tests sets with size 0.25. Create linear regressor. Train regressor with training data. Print out R squared with function print_metrics.
3. Use backward elimination for feature selection. Create training and test sets with selected features. Call linear regression function created in step 5 with selected features.
4. Create polynomial regressor with degree 2. Transform features with polynomial regressor. Call linear regression function created in step 5 with transferred features.
5. Scale features and dependent variable with Standard scaler. Split dataset into training and tests sets with size 0.25. Create SVR regressor with rbf kernel and auto gamma. Train regressor with training set. Print out R squared with function print_metrics.
6. Split dataset into training and tests sets with size 0.25. Create decision tree regressor. Train regressor with training set. Print out R squared with function print_metrics.
7. Split dataset into training and tests sets with size 0.25. Create random forest regressor with 10 estimators. Train regressor with training set. Print out R squared with function print_metrics.

## LAB 3 - Dataset preprocessing, classification models trials and results analysis
**NB! Dataset that was used in this lab is private, so students.xlsx was replaced with a blank file.**
### The purpose of this assignment
The goal of this lab is to check that the student has knowledge in the following topics:
* Dataset preprocessing.
* Logistic regression.
* K-nearest neighbours.
* Support vector machine classification.
* Naive Bayes.
* Decision tree classification.
* Random forest classification.

### Business problem description
```
There is need to develop machine learning model to predict is student likely to not continue studies
aﬅer 4th semester based on collected data.
Goal is to predict students not likely to continue studies. 
This introduces constraints with model "accuracy" because most students will continue studies and 
there can be additional not known factors contributing to decision not to continue. 
Goal can be achieved to find out model with minimum amount of false positives. 
This can make model too pessimistic and most likely to predict many false negatives.
```

### Task
1. Read dataset file to pandas data frame.
2. Save dataset description and categorical features to file in results directory.
3. Filter out students not continuing studies (In university aﬅer 4 semesters is No) and save filtered description and categorical features to file.
4. By comparing and studying results saved in steps 4 and 5 try to get familiar with data.
5. Preprocess data by encoding categorical features and by replacing missing numeric exam points with 0.
6. Create logistic regressor. Split test and training data (0.25 is default). Scale features. Train regressor. Predict test set results and print out metrics.
7. Create K-nn classifier. Split test and training data (0.25 is default). Scale features. Train classifier. Predict test set results and print out metrics.
8. Create SVM classifier - SVC. Split test and training data (0.25 is default). Scale features. Train classifier. Predict test set results and print out metrics.
9. Create Naive Bayes classifier. Split test and training data (0.25 is default). Scale features. Train classifier. Predict test set results and print out metrics.
10. Create decision tree classifier. Split test and training data (0.25 is default). Scale features. Train classifier. Predict test set results and print out metrics.
11. Create Random forest classifier. Split test and training data (0.25 is default). Scale features. Train classifier. Predict test set results and print out metrics.
12. Select most suitable classifier to solve business problem and submit your selection with rationale as answer.

### Answer
```
"Goal can be achieved to find out model with minimum amount of false positives."
As I have the lowest number of false-positive with Naive-Bayes (ComplementNB), 
I suppose it suits the best here.
Maybe K-NN is the best though, since it has a bit bigger amount of false-positives 
but less false-negatives. 
Plus, on your example dataset printouts example K-NN was better than Naive-Bayes.
```

## LAB 4 - Clustering trials and multidimensional features visualisation
### The purpose of this assignment
The goal of this lab is to check that the student has knowledge in the following topics:
* Dataset preprocessing.
* k-means clustering.
* Density-based spatial clustering of applications with noise (DBSCAN).
* Multidimensional features visualisation.

### Business problem description
```
HELP International have been able to raise around $10 million. 
Now the CEO of the NGO needs to decide how to use this money strategically and effectively. 
So, CEO has to make decision to choose the countries that are in the direst need of aid. 
Hence, your Job as a Data scientist is to categorise the countries using some socio-economic and 
health factors that determine the overall development of the country. 
Then you need to suggest the countries which the CEO needs to focus on the most.
```

### Task
1. Read dataset file to pandas data frame.
2. Save dataset description to file in results directory.
3. Preprocess dataset if needed.
4. Find possible suitable number of clusters with help of elbow method. WCSS plot shall be saved to results folder for review.
5. Visualise dataset with help of t-SNE dimensions reduction to 2 dimensions.
6. Select suitable clustering algorithm for your business problem and data set.
7. Find clusters.
8. Visualise dataset with found clusters with help of t-SNE dimensions reduction by adding different colour and symbol to each cluster.

### Dataset
https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data

### Answer
Results
```
         cluster  child_mort   exports         gdpp   health   imports       income  inflation  life_expec  total_fer 
         0          5.000000 58.738889 42494.444444 8.807778 51.491667 45672.222222   2.671250   80.127778   1.752778
         1         92.961702 29.151277  1922.382979 6.388511 42.323404  3942.404255  12.019681   59.187234   5.008085
         2         21.927381 40.243917  6486.452381 6.200952 47.473404 12305.595238   7.600905   72.814286   2.307500

From the table above, we can see that the most affected countries are from the cluster 1.
```