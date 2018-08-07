#Setp1: 导入相关的包

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Setp2: 读取数据

data = pd.read_csv('E:\MORTAL-IT\python-projects\ml\Linear_Regression\ML\Logistic_Regression\data\Socials.csv')
# 获取第二列和第三列的数据
X = data.iloc[:, [2, 3]].values
# 
y = data.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

### Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 3 建立逻辑回归模型并训练模型

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

## Step 4 模型预测

y_pred = classifier.predict(X_test)

# Step 5 评估模型

### Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Setp 6 可视化
# plt.scatter(y_test , y_pred, color = 'red')
plt.plot(X_train , classifier.predict(X_train), color ='blue')
plt.plot(y_test , y_pred, color ='red')
plt.plot(X_train , y_train, color ='red')