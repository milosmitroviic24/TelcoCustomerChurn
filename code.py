import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging

# Loading data
dataframe = pd.read_csv('churn-bigml-80.csv')
print(dataframe.head())
# Data loaded

# Missing values?
numOfMissingValues = dataframe.isnull().sum()
print(numOfMissingValues)
# There are no missing values in the loaded data

# Outliers?
numericCols = dataframe.select_dtypes(include = ['Int64', 'Float64']).columns
percentageToRemove = 0.10
for numericCol in numericCols:
    Q1 = dataframe[numericCol].quantile(0.25)
    Q3 = dataframe[numericCol].quantile(0.75)
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    isOutlier = (dataframe[numericCol] < lowerBound) | (dataframe[numericCol] > upperBound)
    outliers = dataframe[isOutlier].copy()
    nToRemove = int(len(outliers) * percentageToRemove)
    if nToRemove > 0:
        outliers['distance'] = (outliers[numericCol] - outliers[numericCol].median()).abs()
        toRemove = outliers.sort_values(by = 'distance', ascending = False).head(nToRemove).index
        dataframe.drop(index = toRemove, inplace = True)
    dataframe[numericCol] = dataframe[numericCol].clip(lower = lowerBound, upper = upperBound)

# EDA
stateChurnCount = dataframe.groupby(['State', 'Churn']).size().unstack(fill_value = 0)
print(stateChurnCount)

areaCodeChurnCount = dataframe.groupby(['Area code', 'Churn']).size().unstack(fill_value = 0)
print(areaCodeChurnCount)

df_mi = dataframe.copy()
le = LabelEncoder()
categoricalCols = ['Area code', 'State']
for col in categoricalCols:
    df_mi[col] = LabelEncoder().fit_transform(df_mi[col])
miScores = mutual_info_classif(df_mi[categoricalCols], df_mi['Churn'], discrete_features = True)
for col, score in zip(categoricalCols, miScores):
    print(f"{col}: {score:.4f}")

dataframe = dataframe.drop(['State'], axis = 1, errors = 'ignore')
dataframe = dataframe.drop(['Area code'], axis = 1, errors = 'ignore')
print(dataframe.head())

# One-Hot Encoding
catCols = dataframe.select_dtypes(include = ['object', 'category']).columns
ohe = OneHotEncoder(drop = 'first', sparse_output = False)
encodedArray = ohe.fit_transform(dataframe[catCols])
encodedCols = ohe.get_feature_names_out(catCols)
encodedDataFrame = pd.DataFrame(encodedArray, columns = encodedCols, index = dataframe.index)
dataframe = pd.concat([dataframe.drop(catCols, axis = 1, errors = 'ignore'), encodedDataFrame], axis = 1)

print(dataframe.head())
# Categorical variables encoded

# EDA - Visualization
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Churn', y = 'Customer service calls', data = dataframe)
plt.title('Distribution Customer service calls - Churn')
plt.xlabel('Churn')
plt.ylabel('Customer service calls')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Churn', y = 'Account length', data = dataframe)
plt.title('Boxplot - Account length vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Account length')
plt.grid(True)
plt.show()

accountLengthMutInfo = mutual_info_classif(dataframe[['Account length']], dataframe['Churn'], discrete_features = True)
print("Mutual information for Account length: ", accountLengthMutInfo)

correlationMatrix = dataframe.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(correlationMatrix, annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title("Correlation Matrix")
plt.show()

dataframe = dataframe.drop(['Voice mail plan_Yes'], axis = 1, errors = 'ignore')
print(dataframe.head())

internationalPlanInfo = mutual_info_classif(dataframe[['International plan_Yes']], dataframe['Churn'], discrete_features = True)
print("Mutual information for International plan: ", internationalPlanInfo)

churnCounts = dataframe['Churn'].value_counts()
plt.figure(figsize = (8, 6))
plt.pie(churnCounts, labels = ['No Churn', 'Churn'], autopct = '%1.1f%%', startangle = 90, colors = ['#66b3ff', '#ff9999'])
plt.title('Pie - Churn')
plt.show()

X = dataframe.drop(['Churn'], axis = 1)
y = dataframe['Churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balancing
smt = SMOTETomek(random_state = 42)
X_resampled, y_resampled = smt.fit_resample(X_scaled, y)

churnCountsResampled = y_resampled.value_counts()
plt.figure(figsize = (8, 6))
plt.pie(churnCountsResampled, labels = ['No Churn', 'Churn'], autopct = '%1.1f%%', startangle = 90, colors = ['#66b3ff', '#ff9999'])
plt.title('Pie - Churn - Resampled')
plt.show()

# Train - test splitting
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42, stratify = y_resampled)

# Optimal K?
def plot_elbow_optimal_K(X_train, y_train, kRange = [1, 3, 5, 7, 9, 11], cvFolds = 5):
    avgErrors = []
    for k in kRange:
        cv = StratifiedKFold(n_splits = cvFolds, shuffle = True, random_state = 42)
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv = cv, scoring = 'accuracy')
        avgErrors.append(1 - np.mean(scores))
    plt.figure(figsize = (8, 6))
    plt.plot(kRange, avgErrors, marker = 'o', color = 'b')
    plt.xlabel('K')
    plt.ylabel('Average error')
    plt.title('Elbow method for optimal K (KNN)')
    plt.xticks(kRange)
    plt.show()

plot_elbow_optimal_K(X_train, y_train)
optimalK = 7 # Recognized by plotted graph (elbow method)

baseModelsStacking = [
    ('knn', KNeighborsClassifier(n_neighbors = optimalK)),
    ('dt', DecisionTreeClassifier(criterion = 'gini'))
]

metaLearnerStacking = LogisticRegression()

hyperparamsStacking = {
    'final_estimator__C': [0.01, 0.1, 1]
}

hyperparamsRandomForest = {
    'n_estimators' : [50, 60, 70],
    'max_depth' : [3, 4]
}

hyperparamsXGB = {
    'n_estimators' : [50, 60, 70],
    'learning_rate' : [0.04, 0.06],
    'max_depth': [2, 3]
}

hyperparamsLightGBM = {
    'num_leaves': [3, 4, 5],
    'feature_fraction' : [0.65, 0.70],
    'subsample' : [0.7, 0.8]
}

modelsHyperparams = [
    (StackingClassifier(estimators = baseModelsStacking, final_estimator = metaLearnerStacking), hyperparamsStacking),
    (RandomForestClassifier(criterion = 'entropy', random_state = 42), hyperparamsRandomForest), # Bagging
    (XGBClassifier(random_state = 42), hyperparamsXGB),
    (LGBMClassifier(n_estimators = 100, random_state = 42), hyperparamsLightGBM)
]

def find_optimal_model(modelsHyperparams):
    optimalStacking, optimalRandomForest, optimalXGB, optimalLGB = None, None, None, None
    for model, hyperparams in modelsHyperparams:
        if model.__class__.__name__ == 'StackingClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv = 5)
            gridSearch.fit(X_train, y_train)
            optimalStacking = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by = 'mean_test_score', ascending = False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'RandomForestClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv = 5)
            gridSearch.fit(X_train, y_train)
            optimalRandomForest = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'XGBClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv = 5)
            gridSearch.fit(X_train, y_train)
            optimalXGB = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'LGBMClassifier':
            logging.basicConfig(level=logging.ERROR)
            gridSearch = GridSearchCV(model, hyperparams, cv = 5)
            gridSearch.fit(X_train, y_train)
            optimalLGB = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        else:
            print("Model not recognized!")
    return [optimalStacking, optimalRandomForest, optimalXGB, optimalLGB]

optimalModels = find_optimal_model(modelsHyperparams)

# Feature selection
rfe_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nFeatures in range(1, len(X.columns) + 1):
    rfe = RFE(estimator = RandomForestClassifier(n_estimators = 60, max_depth =  4, criterion = 'entropy' , random_state = 42), n_features_to_select = nFeatures)
    scores = cross_val_score(rfe, X_train, y_train, cv = cv, scoring = 'accuracy')
    rfe_scores.append(np.mean(scores))

numFeaturesRange = range(1, len(X.columns) + 1)
numFeatures = list(numFeaturesRange)

plt.figure(figsize = (8, 6))
plt.plot(numFeatures, rfe_scores, marker = 'o', color = 'r')
plt.xlabel('Number of (selected) features')
plt.ylabel('Mean accuracy')
plt.title('Elbow method for optimal number of (selected) features')
plt.show()

optimalNumFeatures = 8
finalRFE = RFE(RandomForestClassifier(n_estimators = 60, max_depth = 4, criterion = 'entropy', random_state = 42), n_features_to_select = optimalNumFeatures)
finalRFE.fit(X_train, y_train)
original_column_names = X.columns.tolist()
X_train = pd.DataFrame(X_train, columns=original_column_names)
X_test = pd.DataFrame(X_test, columns = original_column_names)
selectedFeatures = X_train.columns[finalRFE.support_].tolist()
print("Selected features: ", selectedFeatures)

def find_optimal_models_selected_features(modelsHyperparams):
    optimalStacking, optimalRandomForest, optimalXGBoost, optimalLightGBM = None, None, None, None
    X_train_SF = X_train[selectedFeatures]
    for model, hyperparams in modelsHyperparams:
        if model.__class__.__name__ == 'StackingClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv = 5)
            gridSearch.fit(X_train_SF, y_train)
            optimalStacking = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'RandomForestClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv=5)
            gridSearch.fit(X_train_SF, y_train)
            optimalRandomForest = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'XGBClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv=5)
            gridSearch.fit(X_train_SF, y_train)
            optimalXGBoost = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        elif model.__class__.__name__ == 'LGBMClassifier':
            gridSearch = GridSearchCV(model, hyperparams, cv=5)
            gridSearch.fit(X_train_SF, y_train)
            optimalLightGBM = gridSearch.best_estimator_
            cv_results_df = pd.DataFrame(gridSearch.cv_results_)
            cv_results_df = cv_results_df.sort_values(by='mean_test_score', ascending=False)
            print(cv_results_df[['params', 'mean_test_score']])
        else:
            print("Model not recognized.")
    return [optimalStacking, optimalRandomForest, optimalXGBoost, optimalLightGBM]

optimalModelsSelectedFeatures = find_optimal_models_selected_features(modelsHyperparams)

def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    rocAuc = auc(fpr, tpr)
    plt.figure(figsize = (10, 6))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC Curve (area = {rocAuc:.2f})')
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve - {model.__class__.__name__}')
    plt.legend(loc = 'lower right')
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confMatrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, confMatrix

print("Scores for optimal models (all features):")
for optimalModel in optimalModels:
    name = optimalModel.__class__.__name__
    accuracy, precision, recall, f1, confMatrix = evaluate_model(optimalModel, X_test, y_test)
    print(f"{name} - metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"Confusion matrix:\n{confMatrix}")
    plot_roc_curve(optimalModel, X_test, y_test)

print("Scores for optimal models (selected features):")
for optimalModelSF in optimalModelsSelectedFeatures:
    name = optimalModelSF.__class__.__name__
    accuracy, precision, recall, f1, confMatrix = evaluate_model(optimalModelSF, X_test[selectedFeatures], y_test)
    print(f"{name} - metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"Confusion matrix:\n{confMatrix}")
    plot_roc_curve(optimalModelSF, X_test[selectedFeatures], y_test)