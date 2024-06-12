import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 导入数据
df_train = pd.read_csv("./train.csv", index_col="PassengerId")
df_test = pd.read_csv("./test.csv", index_col="PassengerId")
# 合并数据集
df_all = pd.concat([df_train, df_test])
df_all.info()
# 查看缺失值
print("\n1、查看数据并处理缺失值...")
print(df_all.isnull().sum())
# Age缺失值处理
df_all["Age"] = df_all.groupby(["Sex", "Pclass"])["Age"].apply(lambda x: x.fillna(x.median()))
# Embarked缺失值处理
df_all["Embarked"] = df_all["Embarked"].fillna("S")
# Fare缺失值处理
med_fare = df_all.groupby(["Pclass", "Parch", "SibSp"]).Fare.median()[3][0][0]
df_all["Fare"] = df_all["Fare"].fillna(med_fare)
# Cabin缺失值处理
df_all["Deck"] = df_all["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "M")
gp = df_all.groupby(by=["Deck", "Pclass"])["Pclass"]
gp.count()
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df_all.loc[df_all['Deck'] == 'T', 'Deck'] = 'A'
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
df_all.drop(['Cabin'], inplace=True, axis=1)
print("\n2、缺失值处理完毕!")
print(df_all.isnull().sum())
# 分析目标变量分布情况
n_train = 0
for i in df_train["Survived"]:
    n_train += 1
not_survived, survived = df_train["Survived"].value_counts()
print("\n3、目标变量分布情况：")
print("Survived passengers:{:.2f}%".format(survived / n_train))
print("Not survived passengers:{:.2f}%".format(not_survived / n_train))
# 相关分析：统计高相关，绘制相关系数图
df_all_corr = df_all.corr().abs()
df_all_corr_sort = df_all_corr.unstack().sort_values(ascending=False).reset_index()
dd1 = df_all_corr_sort.iloc[:, 2] > 0.1
dd2 = df_all_corr_sort.iloc[:, 2] < 1
hight_corr = df_all_corr_sort[dd1 & dd2]
print("\n4、相关分析：")
print(hight_corr)
sns.heatmap(df_all_corr, annot=True, square=True, cmap="coolwarm")


# 连续特征的目标分布
def traverse_show(cont_features):
    for feature in cont_features:
        surv = (df_train['Survived'] == 1)
        sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True)
        sns.distplot(df_train[surv][feature], label='Survived', hist=True)
        plt.legend(loc='upper right')
        plt.show()


traverse_show(cont_features=['Age', 'Fare'])
traverse_show(cont_features=['Fare', 'Age'])
# # 类别特性的目标分布
# cat_features = ["Embarked", "Parch", "Pclass", "Sex", "SibSp", "Deck"]
# for feature in cat_features:
#     sns.countplot(x=feature, hue="Survived", data=df_train)
#     plt.xlabel(feature)
#     plt.ylabel("Passenger Count")
#     plt.legend(["Not Survived", "Survived"], loc="upper center")
#     plt.title("Survived in {} Feature".format(feature))
#     plt.show()
# Fare连续特性分段
df_all["Fare"] = pd.qcut(df_all["Fare"], 13)
plt.figure(figsize=(22, 9))
sns.countplot(x="Fare", hue="Survived", data=df_all)
plt.xlabel("Fare")
plt.ylabel("Passenger Count")
plt.legend(["Not Survived", "Survived"], loc="upper right")
plt.title("Count of Survived in Fair Feature")
plt.show()
# Age连续性特性分段
df_all["Age"] = pd.qcut(df_all["Age"], 10)
plt.figure(figsize=(22, 9))
sns.countplot(x="Age", hue="Survived", data=df_all)
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.legend(["Not Survived", "Survived"], loc="upper right")
plt.title("Count of Survived in Age Feature")
plt.show()
# 编码频数特征
df_all["Family_size"] = df_all["SibSp"] + df_all["Parch"] + 1
sns.countplot(x="Family_size", hue="Survived", data=df_all)
plt.legend(["Not Survived", "Survived"], loc="upper right")
plt.title("Count of Survived in Family size")
plt.show()
family_map = {1: "Alone", 2: "Small", 3: "Small", 4: "Small", 5: "Medium", 6: "Medium", 7: "Large", 8: "Large",
              11: "Large"}
df_all["Family_size_Grouped"] = df_all["Family_size"].map(family_map)
sns.countplot(x="Family_size_Grouped", hue="Survived", data=df_all)
plt.legend(["Not Survived", "Survived"], loc="upper right")
plt.title("Count of Survived in Family size After Grouped")
plt.show()
# Ticket按频数编码
df_all["Ticket_Frequency"] = df_all.groupby("Ticket")["Ticket"].transform("count")
sns.countplot(x="Ticket_Frequency", hue="Survived", data=df_all)
plt.xlabel("Ticket Frequency")
plt.ylabel("Passenger Count")
plt.legend(["Not Survived", "Survived"], loc="upper right")
plt.title("Count of Survived in Ticket_Frequency Feature")
plt.show()
# # Title Married按类别编码
# df_all["Title"] = df_all["Name"].str.split("，", expand=True)[1].str.split(".", expand=True)[0]
# df_all["Married"] = 0
# df_all[" Married"].loc[df_all["Title"] == "Mrs"] = 1
# df_all["Title"] = df_all["Title"].replace(["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the countess", "Dona"], "Ms")
# df_all["Title"] = df_all["Title"].replace(["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"], "Sir")
# sns.barplot(x=df_all["Title"].value_counts().index, y=df_all["Title"].value_counts().values)
# plt.title("Title Feature Value Counts After Grouping")
# plt.show()
# # 类别标签编码非数值特性
# non_numeric_features = ["Embarked", "Sex", "Deck", "Title", "Family_Size_Grouped", "Age", "Fare"]
# for feature in non_numeric_features:
#     df_all[feature] = LabelEncoder().fit_transform(df_all[feature])
# # 独热编码类型特性
# cat_features = ["Pclass", "Deck", "Embarked", "Title", "Family_Size_Grouped"]
# encoded_features = []
# for feature in cat_features:
#     encoded_feat = OneHotEncoder().fit_transform(df_all[feature].values.reshape(-1, 1)).toarray()
#     n = df_all[feature].nunique()
#     cols = ["{}_{}".format(feature, n) for n in range(1, n + 1)]
#     encoded_df = pd.DataFrame(encoded_feat, columns=cols)
#     encoded_df.index = df_all.index
#     encoded_features.append(encoded_df)
#     df_all = pd.concat([df_all, encoded_features[-1]], axis=1)
# 4.模型
# 训练集和测试集
df_train = df_all.iloc[:n_train]
y_train = df_train["Survived"]
x_train = df_train.drop(columns=["Survived"])
x_test = df_all.iloc[n_train:].drop(["Survived"], axis=1)
y_test = pd.read_csv("./gender_submission.csv", index_col="PassengerId")
# 模型参数优化
param = {
    "n_estimators": range(1000, 1200, 100),
    "max_depth": range(2, 20, 1),
    "min_samples_split": [4],
    "min_samples_leaf": [5]
}
rfc = RandomForestClassifier(criterion='gini ', oob_score=True, random_state=42)
gs = GridSearchCV(rfc, param, cv=3)
gs.fit(x_train, y_train)
print(gs.best_params_, gs.best_score_)
# 预测评估
gs.best_estimator_.predict(x_test)
score = gs.best_estimator_.score(x_test, y_test)
print(score)
# 显示特征重要性
importances = gs.best_estimator_.feature_importances_
names = x_test.columns.tolist()
feature_importantes = pd.DataFrame(importances, index=names, columns=["importances"])
feature_importantes.sort_values(by="importances", inplace=True, ascending=False)
plt.figure(figsize=(9, 12))
sns.barplot(x="importances", y=feature_importantes.index, data=feature_importantes)
plt.title("Random Forest Classifier Feature Importance")
plt.show()
