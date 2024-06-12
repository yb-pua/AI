from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 让汉字正常显示
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_csv('Advertising.csv')
x = df[['TV', 'radio', 'newspaper']]
y = df['sales']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)
lm = LinearRegression()
lm.fit(x_train, y_train)

y_pred1 = lm.predict(x_train)
R2 = r2_score(y_train, y_pred1)
print("R² --> {}".format(R2))

y_pred2 = lm.predict(x_test)
plt.plot(range(len(y_pred2)), y_pred2, c='g', label='Predict')
plt.plot(range(len(y_pred2)), y_test, c='r', label='Test(label)')
plt.legend(loc='lower left')
plt.xlabel("销售数量")
plt.ylabel("销售价值")
plt.title('线性分析销售额与广告投入关系')
plt.show()
