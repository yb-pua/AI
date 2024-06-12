from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 导入数据去除性别列选择需要的x和y数据
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:,4].values
# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 使用K-NN对训练集数据进行训练
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
param_name = 'n_neighbors'
param_range = range(1, 51)
# scoring: 分类用 accuracy, 回归用 mean_squared_error
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, cv=5,
    param_name=param_name, param_range=param_range,scoring='accuracy')
# 交叉验证训练集得分
train_scores_mean = np.mean(train_scores, axis=1)
# 交叉验证测试集得分
test_scores_mean = np.mean(test_scores, axis=1)
# 最优K值可视化
plt.plot(param_range,train_scores_mean, color='red', label='train')
plt.plot(param_range, test_scores_mean, color='green', label='test')
plt.legend('best')
plt.xlabel('param range of k')
plt.ylabel('scores mean')
plt.show()
# 使用最优K值做模型训练预测
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
# 预测准确率得分
score = classifier.score(X_test, y_test)
print(score)

