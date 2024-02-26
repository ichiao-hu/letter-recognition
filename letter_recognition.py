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
sns.set_style("whitegrid")

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
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
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
