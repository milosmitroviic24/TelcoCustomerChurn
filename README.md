# Telco Customer Churn (Predictor)

## Description of the problem

Based on characteristics such as 
- State
- Account length
- Area code
- International plan
- Voice mail plan
- Number vmail messages
- Total day minutes
- Total day calls
- Total day charge
- Total eve minutes
- Total eve calls
- Total eve charge
- Total night minutes
- Total night calls
- Total night charge
- Total intl minutes
- Total intl calls
- Total intl charge
- Customer service calls
train a model to predict whether the user will continue to use the services of the telecommunications company or not.

## Data loading

Firstly, we load the data that is located in the CSV file named 'churn-bigml-80.csv'. We do this with the help of the Pandas library which makes it easier for us to work with data in terms of analysis, manipulation and other operations on the same.

## Data preprocessing

### Checking for missing values

We create a temporary "numOfMissingValues" in which we place the number of missing fields over the created dataframe (tabular object from the Pandas package into which we loaded the data from the CSV file). 
As we have no missing values in the dataframe, we have no obligation to fill empty fields with average, the most frequent or other statistical values by columns.

Supporting packages: Pandas

### Handling outliers

The IQR method was used to handle outliers. Outliers are registered for each of the attributes and for each attribute 10% of the most critical are removed, and the rest are moved to near limit.

Supporting packages: Pandas 

### Encoding categorical columns

One-Hot Encoding method was used for encoding categorical columns. The categorical attribute is divided into (n-1) columns (n - the number of categories for the column), where as a rule only one of the newly created columns can have the value 1, all the others 0. We do not have to use all n because if we have the value 0 on all newly created (n-1), we know that the value of the attribute for the sample is the one remaining.

Supporting packages: sklearn.preprocessing

### Review of the first 5 rows

print (dataframe.head()) 

We do this just to see what is done with dataframe until now.

### Exploratory Data Analysis

#### Box plot: Customer service calls - Churn

The goal of this box plot, ie. of this visualization is to analyze the median, Q1, Q3, whiskers and outliers for Churn = 0, ie. Churn = 1.
The motive is the assumption - a greater number of 'Customer service calls' potentially indicates a greater number of service problems encountered by the user, which accumulates more and more user dissatisfaction and increases the chance that user will stop using the services of the telecommunications company.
Box plots have a common area, so - they do not indicate the realization of the assumption, but they are far from being very similar. The attribute is definitely worth keeping.

Supporting packages: seaborn, matplotlib

#### Distribution of Churn values in relation to State and Area code values 

The number of occurrences Churn = 1 and Churn = 0 in relation to the values of the Area code attribute is monitored. It is done the same for the State attribute. Nothing more concrete could be concluded from the published results, so a Mutual Info analysis was performed for both attributes in relation to the target variable. Mutual Info analysis confirmed that no conclusion can be drawn from these attributes for predicting the target variable, so it is legitimate to remove them, because they can potentially introduce unnecessary additional complexity into the data.

Supporting packages: sklearn.feature_selection

#### Box plot: Account length - Churn

It is logical to consider the influence of the variable Account length, following the assumption that users who have been in cooperation with a telecommunications company for a longer period of time are more loyal to the same - and there is less chance that they will terminate the cooperation (Churn = 1).
The analysis tool is the box plot, which indicated that there is almost no difference between the box plot (Churn = 1) and the box plot (Churn = 0). 
Therefore, it is not realistic to make any conclusion about the prediction based solely on the value of this variable. This is supported by the very low Mutual Info value between this variable and the target variable.
However, the following real situations should also be taken into account, regarding the influence of this variable in combination with the variables that talk about the number of minutes, the number of calls, the total cost. For example,  users with a large value of Account length and a small number of calls or users with a small account length and high costs can potentially be dissatisfied and consider the option of leaving the services of the telecommunications company.
This is supported by the values from the correlation matrix.

Supporting packages: sklearn.feature_selection, seaborn, matplotlib

#### Voice mail plan - Necessary or not?

It is interesting to analyze the relationship between the variables Voice mail plan and Number vmail messages.
From the correlation matrix it can be read that these variables are highly correlated, so we can remove one of them, because we conclude that those users who do not have the right to voice messages cannot use them, while those who do have a non-zero value for the number vmail messages variable. 
Therefore, users with Voice mail plan = Yes, have non-zero values, while users with Voice mail plan = No, have zero values, so there is no point in keeping both variables, and we choose to keep the one that carries stronger information, which is Number vmail messages.

Supporting packages: sklearn.feature_selection, seaborn, matplotlib

#### International plan - Necessary or not?

It is important to take into account the influence of the International plan variable.
Mutual info was calculated for this variable in relation to the target variable. The weak value of Mutual Info, however, should not mislead, but the relationship of this variable with the variables concerning international calls, minutes and costs should be considered, which is meaningful. It may happen that the weak correlation between these variables means the following - the user has international services turned on, but does not register benefits after international calls compared to those users who do not have international services turned on. This can lead to potential dissatisfaction (Churn = 1).

Supporting packages: sklearn.feature_selection, seaborn, matplotlib

### Data scaling 

In a dataset, some columns may have data on a larger scale than some other columns. This can have a sensitive (and inconvenient) impact on models such as KNN, SVM and others. For example, with KNN where distances are calculated, an unscaled dataset can be misleading and lead to the model not learning basic patterns. When using models of this type, it is important to reduce the attributes to approximate weight scales.

Supporting packages: sklearn.preprocessing

### Class balancing

Now, after the set of attributes has passed the primary cut and the data are scaled, and before that the problem of categorical variables, i.e. missing values and anomalies has been regulated, it is necessary to check the balance of the classes where the aim is to have as much balance as possible, so that the model is not subordinated to more numerous voices. SMOTETomek technique was used, where SMOTE component represents the generation of synthetic samples of the minority class, and the Tomek technique cleanes - deteles the samples of the majority class, which are spatially closest to the samples of the minority class.

Supporting packages: sklearn.preprocessing

Before balancing: 
* Churn = 0: 85.9%
* Churn = 1: 14.1%
After balancing:
* Churn = 0: 50.0%
* Churn = 1: 50.0%

Supporting packages: imblearn.combine

## Dataset = Training set + Test set

Division of the scaled, balanced dataset into training and test set, in the ratio 80-20 [%], where the balance of the classes is preserved both in the training set and in the test set.

Supporting packages: sklearn.model_selection

## Models Training

Three techniques were used to train the models:
* Stacking
* Bagging (Random Forest)
* Gradient Boosting (XGBoost, LightGBM)

### a) Stacking 

As part of the Stacking technique, KNN and Decision Tree were used as base models.
KNN models, when created, require a clearly set parameter n_neighbors. 
In order to choose the best possible value for that parameter, the elbow method is used, where for different values of K (n_neighbors) the values of the average error of KNN models trained with those values are monitored.
Here, cross-validation is used to calculate the average accuracy (average error = 1 - average accuracy). For pair corresponding values of K - Average error, a line graph is drawn, and the K - elbow coordinate is optimum for n_neighbors.
The elbow is a point, for which it is important that with further increase in K, the error decreases slightly.
In this case, the K value for elbow is 5.
For Decision Tree, as a base model, only the node purity criteria parameter was set - it was taken to be gini impurity.

KNN (with the passed optimal parameter n_neighbors) and Decision Tree are trained separately on the training set (its subset strictly used for training the base models). After that, the meta-learner (in this case - Logistic Regression) is trained on the remaining part of the training set (the second subset, complementary to the mentioned one).
The meta-learner is taught to combine the predictions of the base models on the second subset (that subset was not seen by the base models before, but is like a validation set).

The hyperparameter that is being optimized in the Stacking model is C regularization parameter for the meta-learner.
C by definition represents the reciprocal value of the penalization coefficient.

Supporting packages: sklearn.neighbors, sklearn.tree, sklearn.linear_model, sklearn.ensemble

### b) Bagging

Within the Bagging technique of model training, Random Forest is used. 
Bagging = Bootstrap + Aggregating
Random Forest is an ensemble technique - it includes multiple decision trees, which are trained independently and parallel to each other.
Each decision tree is trained on its own training set, which has the same number of training samples as the general training set.
Samples for training trees are chosen randomly and with repetition.
This marks the bootstrap component of bagging.
The hyperparameters that are optimized refer to the maximum allowed tree depth and the number of trees.
When the models are finished training and testing is done on unseen data, the final prediction of the Bagging model is obtained by majority vote.
This marks the aggregating component of bagging.

Supporting packages: sklearn.ensemble

### c) Gradient Boosting

#### c.1.) XGBoost

XGBoost is a specific type of gradient boosting technique, where each model in the ensemble, excpet the first one that is trained to predict the target variable, is trained to correct the errors of the previous model.
Specific features of this technique:
* parallelization at the node level
* sparsity aware (automatic handling of missing values)
* early stopping rounds
* etc.

The hyperparameters that we optimize in the XGBoost model are related to the number of estimators, the learning rate and the maximum allowed depth of each tree (estimator).
Typically, if there are more estimators, then the learning rate is lower, and vice versa. This is where balance is aimed - so that the model does not learn errors too quickly (trap for overfitting), nor too slowly (trap for underfitting). 
Also, since there are more weak models, it is not good for them to be excessively complex, but to learn slowly but surely - that is why the depth of weak models is generally limited, where they are not allowed to deepen and 'abruptly' learn the errors of their predecessors.

Supporting packages: xgboost

#### c.2.) LightGBM

LightGBM is a specific type of gradient boosting technique, which is designed to save time and memory.
The time is kept so that the development of the trees takes place according to the leaf-wise principle.
The memory is saved so that not all samples are trained, but the training emphasis is on samples with a larger gradient (larger error), while the others (with a smaller gradient/error) are chosen randomly. This is called Gradient One Side Sampling (GOSS).
Also, the fact that not all features from the training set are included for training each tree, but a defined percentage of them is randomly selected, contributes to saving memory. This is called EFB (Exclusive Feature Bundling).
Discrete binning of numeric attribute values also helps in memory conservation.

The hyperparameters that are optimized in this model refer to the maximum allowed number of leaves (smaller number of leaves - simpler tree, fight against overfitting), the percentage of samples that are taken into consideration for training each tree, as well as the percentage of attributes that are randomly selected for training each tree.

Supporting packages: lightgbm

### Hyperparameterization

Through the function find_optimal_model, both training and hyperparameterization of the model are performed.
What happens?
For the passed model and passed hyperparameters, the GridSearchCV object trains the given model for each combination of the passed hyperparameters, including cross-validation, where based on the results of its metrics, the combination of hyperparameters that gave the best results for that metric is selected.
This process is performed on each of the four models, and we get all four models with optimally tuned hyperparameters.

Supporting packages: sklearn.model_selection

## Feature Selection

After the models have been trained with all the tuned hyperparameters, where the results are satisfactory, an additional possibility of simplifying the model, based on feature selection, is analyzed.
There is a mechanism for selecting the most influential features on the target variable and its name is RFE (Recursive Feature Elimination).
The mechanism needs to be passed the number of attributes that are intended to remain after selection, where the results are then examined for each possible attribute combination for that number of attributes, and RFE keeps the combination with the best result.
Since it is not known in advance what number of attributes to keep, it is convenient to rely on the elbow method, where for each number of attributes (from 1 to the total number of attributes without selection) the best combinations are examined and their results are monitored.
Where there is an elbow, or where with an additional increase in the number of attributes there is no noticeable increase in the qualiry of the results, the coordinate related to the number of attributes is looked at and that number of attributes is kept.

Then, a final RFE object is created, to which that optimal number of attributes is passed.
This RFE object will examine all combinations of 8 attributes and return the optimal one.
RFE uses the Random Forest model because Random Forest models have built-in mechanism for importance of attributes.

Supporting packages: sklearn.feature_selection

## Hyperparameterization of the models 2

Now, when only the most relevant attributes have been extracted, the hyperparameterization of the models is performed again, which are trained only on these selected attributes.

Supporting packages: sklearn.model_selection

## Evaluation

Both for the optimal models trained on the entire set of attributes and for the optimal models trained on teh set of selected attributes, the evaluation parameters are monitored.
Within this area of the code, there is a function for plotting the ROC curve, where for different values of the positive prediction threshold the True Positive Rate and False Positive Rate parameters are monitored, and a curve is formed and its area is calculated (the closer it is to 1, the better the model).
Also, metrics that are used:

* Accuracy - It is desirable that it is as close as possible to 1.

Accuracy is defined as the ratio of the number of successful predictions to the total number of prediction attempts.

* Precision - It is desirable that it is as close as possible to 1.

Precision is defined as the ratio of correct positive predictions to the total number of positive predictions.

* Recall - It is desirable that it is as close as possible to 1.

Recall is defined as the ratio of correct positive predictions to the total number of positive samples.

* F1 score - It is desirable that it is as close as possible to 1.

F1 score is defined as a measure of the balance between recall and precision.

* Confusion matrix

A confusion matrix indicates the number of successful positive, unsuccessful positive, successful negative and unsuccessful negative predictions of a model.