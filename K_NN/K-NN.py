import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 数据预处理
dataset = pd.read_csv('Socials.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# 分割源数据获取训练数据和测试数据集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 获取KNN模型 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# 预测数据
y_pred = classifier.predict(X_test)


## 模型校验
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

