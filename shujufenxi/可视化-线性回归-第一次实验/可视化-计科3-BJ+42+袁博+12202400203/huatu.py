import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 让汉字正常显示
plt.rcParams["axes.unicode_minus"] = False


def scatter(data):
    x = data["Hours"]
    y = data["Scores"]
    plt.scatter(x, y)
    plt.title("时长和成绩散点图")
    plt.xlabel("hours")
    plt.ylabel("scores")
    plt.show()


def hoursBar(data):
    hours = data["Hours"]
    plt.hist(hours, bins=[i for i in range(1, 11)])
    plt.title("时长直方图")
    plt.show()


def scoreBar(data):
    scores = data["Scores"]
    plt.hist(scores, bins=[i for i in range(10, 101, 10)])
    plt.title("成绩直方图")
    plt.show()


def hoursBoxPlot(data):
    hours = data["Hours"]
    plt.boxplot(hours)
    plt.title("时长线箱图")
    plt.show()


def scoresPie(data):
    pie = count(data)
    plt.pie(pie, labels=['A', 'B', 'C', 'D', 'E'], autopct='%.2f%%')
    plt.title("成绩饼图")
    plt.show()


def scoresBar(data):
    y = count(data)
    x = ['A', 'B', 'C', 'D', 'E']
    plt.bar(x, y)
    plt.title("成绩柱状图")
    plt.show()


def count(data):                # 判断ABCDE等级供成绩饼状图、柱状图使用
    scores = data["Scores"]
    level = [0, 0, 0, 0, 0]
    for i in scores:
        if i >= 90:
            level[0] += 1
        if 80 <= i < 90:
            level[1] += 1
        if 70 <= i < 80:
            level[2] += 1
        if 60 <= i < 70:
            level[3] += 1
        if i < 60:
            level[4] += 1
    size = [level[0], level[1], level[2], level[3], level[4]]
    return size


if __name__ == '__main__':
    datas = pd.read_csv("studentscores.csv")
    scatter(datas)
    hoursBar(datas)
    scoreBar(datas)
    hoursBoxPlot(datas)
    scoresPie(datas)
    scoresBar(datas)
