import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

wine_data = pd.read_csv("./wine.data", header=None)
x, y = wine_data.iloc[:, 1:].values, wine_data.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sc = StandardScaler()
# 对原始化数据做标准化处理
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)
# 做PCA降维
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.fit_transform(x_test_std)
# 显示降维后数据
print("降维后训练的数据：" + str(x_train_pca))
print("降维后测试的数据" + str(x_test_pca))
# 用降维后的数据，训练模型，输出预测得分
lr = LogisticRegression()
lr.fit(x_train_pca, y_train)
score = lr.score(x_test_pca, y_test)
print("预测得分：" + str(score))
