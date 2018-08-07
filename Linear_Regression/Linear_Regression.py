# Step 1 : 导入所需的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: 数据处理
dataset = pd.read_csv('data\\scores.csv')
# 对列数据进行切片处理
X = dataset.iloc[ : , : 1 ].values
# 获取第一列的所有值
Y = dataset.iloc[ : , 1 ].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 


# Step 3: 构建一个线性模型并用数据进行训练
# 
# 实例化一个线性模型
regressor = LinearRegression()
# 得到一个符合坐标下降的线性模型
regressor = regressor.fit(X_train, Y_train)

# Step 4: 用线性模型进行预测

Y_pred = regressor.predict(X_test)

# Step 5: 可视化训练结果 
plt.scatter(X_train , Y_train, color = 'red')
plt.xlabel("train result")
plt.title('Linear Regression Create By MORTAL')

plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.xlabel('predict result')
plt.title('Linear Regression Create By MORTAL')

## 可视化测试结果

plt.scatter(X_test , Y_test, color = 'red')
plt.xlabel('test result')
plt.title('Linear Regression Create By MORTAL')

plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.xlabel('test result')
plt.title('Linear Regression Create By MORTAL')