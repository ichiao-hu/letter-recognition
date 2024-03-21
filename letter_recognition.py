from ucimlrepo import fetch_ucirepo

# fetch dataset
letter_recognition = fetch_ucirepo(id=59)

# data (as pandas dataframes)
X = letter_recognition.data.features
y = letter_recognition.data.targets

# metadata
print(letter_recognition.metadata)

# variable information
print(letter_recognition.variables)

import seaborn as sns
sns.set_style('whitegrid')

# check class imbalance
y.groupby('lettr')['lettr'].count().plot(kind='bar', figsize=(12, 8))

import matplotlib.pyplot as plt

# draw distribution plots for all features
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# draw correlation plot
plt.figure(figsize=(16, 12))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.show()

from sklearn.model_selection import train_test_split

# split data into training, validation, and test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=30)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=30)

import numpy as np

# convert column vectors into 1d arrays
y_train = np.ravel(y_train)
y_val = np.ravel(y_val)
y_test = np.ravel(y_test)

from sklearn.linear_model import LogisticRegression

# multinomial logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

accuracy_lr = accuracy_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred, average='weighted')

from sklearn.ensemble import RandomForestClassifier

# random forests
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred)
f1_rf = f1_score(y_test, y_pred, average='weighted')

# plot feature importances
feature_importances = random_forest_model.feature_importances_
sorted_indices = feature_importances.argsort()[::-1]

plt.figure(figsize=(16, 12))
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.show()

from sklearn.svm import SVC

# support vector machines
svm_model = SVC()
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred)
f1_svm = f1_score(y_test, y_pred, average='weighted')

from sklearn.neighbors import KNeighborsClassifier

# k-nearest neighbors
neighbors_range = list(range(1, 21))
accuracy_scores = []

# hyperparameter tuning for k-nn using elbow method
for n_neighbors in neighbors_range:
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(16, 12))
plt.plot(neighbors_range, accuracy_scores, marker='o')
plt.title('Elbow Method for k-NN Hyperparameter Tuning')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
plt.xticks(neighbors_range)
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=4)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred)
f1_knn = f1_score(y_test, y_pred, average='weighted')

from sklearn.neural_network import MLPClassifier

# neural network: multilayer perceptron
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

accuracy_mlp = accuracy_score(y_test, y_pred)
f1_mlp = f1_score(y_test, y_pred, average='weighted')

import pandas as pd

# compare model performances
df_performance = pd.DataFrame({
    'model': ['Multinomial Logistic Regression', 'Random Forests', 'Support Vector Machines',
              'k-Nearest Neighbors', 'Neural Network'],
    'accuracy': [accuracy_lr, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_mlp],
    'f1_score': [f1_lr, f1_rf, f1_svm, f1_knn, f1_mlp]
})

df_performance