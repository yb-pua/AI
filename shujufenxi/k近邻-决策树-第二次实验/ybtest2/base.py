import sklearn.model_selection as ms
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn import datasets
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np

data = datasets.load_breast_cancer()
# 检查威斯康辛州记录的数据集,569个病人的乳腺癌诊断记录,30个维度的生理指标数据
print(data.data.shape)
# 数据集划分
X_train, X_test, y_train, y_test = ms.train_test_split(data.data, data.target, test_size=0.2, random_state=42)
# 使用0.2即20%的数据作为测试集
print(X_train.shape, X_test.shape)
# 构建决策树模型(优化模型)
model = tree.DecisionTreeClassifier().fit(X_train, y_train)
# 乳腺癌数据集较少，所以采用max_depth为1~10，或者1~20
param_grid = {'max_depth':np.arange(1, 20, 1)}
# 网格搜索优化模型
param = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth':range(2,10,1),
    'min_samples_split' :range(2,5),
    'max_features' :['sqrt', 'log2'],
    'min_impurity_decrease':np.arange(0.1,0.5,0.1)}
clf = tree.DecisionTreeClassifier(random_state=0)
gs = GridSearchCV(clf, param, scoring='accuracy', cv=5).fit(X_train, y_train)
print(gs.best_estimator_, gs.best_score_)
#可视化
n_features = data.data.shape[1]

plt.barh(range(n_features), model.feature_importances_, align='center')
#前者代表y坐标轴的各个刻度，后者代表各个刻度位置的显示的lable
plt.yticks(np.arange(n_features), data.feature_names)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
"""
乳腺癌的5个最重要指标:
worst concave points
worst perimeter
worst texture
worst radius
mean concave points
"""

