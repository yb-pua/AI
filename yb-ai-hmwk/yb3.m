fig3=figure;

Data = [];
%循环取出坐标轴数据
for i = 1:length(lineObj)
    xvalues = get(lineObj(i), 'XData');
    yvalues = get(lineObj(i), 'YData');
    data = [xvalues(:), yvalues(:)];
    Data = [Data; data];
end
%使用 fitgmdist 函数将数据拟合为具有 5 个分量的 GMM 模型。
gmmModel = fitgmdist(Data, 5); 
%使用 linspace 函数定义 x 和 y 的坐标向量，并使用 meshgrid 生成对应的网格点坐标矩阵 X 和 Y。
x = linspace(min(Data(:, 1)), max(Data(:, 1)), 100);
y = linspace(min(Data(:, 2)), max(Data(:, 2)), 100);
[X, Y] = meshgrid(x, y);
%计算 GMM 模型在每个网格点上的概率密度值
Z = pdf(gmmModel, [X(:), Y(:)]);
Z = reshape(Z, size(X));


mesh(Z);
hold on
contour(Z);
Z_shifted = Z + 0.035;

set(findobj(gca,"type","surface"), 'ZData', Z_shifted);
% surf(X, Y, Z);
Z_range = [min(Z_shifted(:)), max(Z_shifted(:))];
clim(Z_range);

azimuth = 45;      
elevation = 25;    
view(azimuth, elevation);

xlabel('X');
ylabel('Y');
zlabel('Z');
