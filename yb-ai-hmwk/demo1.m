fig=openfig('Fig_DataSetM.fig')
lineObj = findobj(fig, 'Type', 'line');
xData = get(lineObj, 'XData');
yData = get(lineObj, 'YData');
color = get(lineObj,'Color');
markerStyle = get(lineObj, 'Marker');

fig1=figure     %创建画布
hold on;   %固定画布
for i=1:5		%循环把5组数据绘制出来
plot(xData{i}, yData{i}, 'LineStyle', 'none', 'Marker', markerStyle{i}, 'MarkerFaceColor', color{i});
end
title('Scatter Plot');
xlabel('X');
ylabel('Y');
%设置图例
legend('Class 1','Class 2','Class 3','Class 4','Class 5','Location','northwest');






