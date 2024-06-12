fig_handle = open('Fig_DataSetM.fig');  % 打开.fig文件，获取图形句柄
plot_children = get(gca, 'Children');  % 获取当前坐标轴的子对象
new_fig = figure;  % 创建一个新的图形窗口
colormap = [0 0 0;0.2 0.2 0.2; 0.4 0.4 0.4;0.6 0.6 0.6;0.8 0.8 0.8];  % 自定义颜色映射
plotmarkers = ['s', 'd', 'v', 'p', 'h','o', 'x', '+', '*', '.'];  % 设定绘制点的标记符号
plota = [];  % 创建一个空变量用于存放绘图数据
data = [];  % 创建一个空变量用于存放数据  
for i = 1:length(plot_children)  % 遍历子对象数组

    childe = plot_children(i);  % 获取当前子对象
    marker = plotmarkers(mod(i-1, length(plotmarkers)) + 1);  % 循环选取标记符号
    if strcmp(get(childe, 'Type'), 'hggroup')  % 如果子对象类型为'hggroup'（图形容器）

        groupn = get(childe, 'Children');  % 获取子对象中的子对象数组
        
    else
        groupn = childe;  % 如果子对象不是'hggroup'类型，则groupn等于该子对象本身
    end
    for j = 1:length(groupn)  % 遍历子对象中的子对象数组
        groupild = groupn(j);  % 获取当前子对象的子对象
        xvalues = get(groupild, 'XData');  % 获取子对象的X坐标数据
        yvalues = get(groupild, 'YData');  % 获取子对象的Y坐标数据
        scatter(xvalues, yvalues, [], colormap(i,:), marker);  % 绘制散点图
        hold on;  % 在同一图形窗口上保持图形
        plota = [plota; [xvalues(:), yvalues(:), repmat(i, length(xvalues), 1)]];  % 将散点图的数据追加到plota变量中
    end
end
title('获取数据');  % 设置图形标题
legend('Location', 'northeast');  % 设置图例的位置
hold off;  % 关闭图形绘制 
data = [data];  %数据赋值data
filename = 'yb-ai-hmwk-test.csv';  %保存文档名
writematrix(data, filename);%写入 
