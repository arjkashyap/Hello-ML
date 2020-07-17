import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

path = "./data/breast-cancer-wisconsin.data"



df = pd.read_csv(path)

df.replace('?',-99999, inplace=True)
df.drop(['id_number'], 1, inplace=True)
df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])

print(df.head())

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("confidence",confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

print(df.head())