# from __future__ import print_function
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# from sklearn import tree
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../final project telegram/new/crop data.csv")

df.head()

x = df.iloc[:,df.columns!='Crop'] #data
y = df.iloc[:,df.columns=='Crop'] #outcome or Label
features = df[['Nitrogen', 'Phosphorous','Potassium','Temperature', 'Humidity', 'Soil pH', 'Rainfall']]
target = df['Crop']
labels = df['Crop']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2)
acc = []
model = []

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(xtrain)
X_test= sc.transform(xtest)

#Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(xtrain,ytrain.values.ravel())
predicted_values = classifier.predict(xtest)
x = metrics.accuracy_score(ytest, predicted_values)
acc.append(x)
model.append('classifier')
print("RF's Accuracy is: ", x)

print(classification_report(ytest,predicted_values))

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
DecisionTree.fit(xtrain,ytrain)
predicted_values = DecisionTree.predict(xtest)

pickle.dump(classifier, open("model.pkl", "wb"))



