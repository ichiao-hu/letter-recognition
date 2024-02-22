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
