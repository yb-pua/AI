Data=[];
for i = 1:length(lineObj)
groupn = lineObj(i);
    for j = 1:length(groupn)
        groupild = groupn(j);
        xvalues = get(groupild, 'XData');
        yvalues = get(groupild, 'YData');
			%循环把数据集合到一个矩阵
        Data = [Data; [xvalues(:), yvalues(:), repmat(i, length(xvalues), 1)]];
    end
end
%分类成3类
new_Data1=Data;
%类别1
for rows=1:size(Data,1)
    if Data(rows,3)~=1&Data(rows,3)~=2&Data(rows,3)~=4
        new_Data1(rows,3)=-1;
    else
        new_Data1(rows,3)=1;
    end
end

new_Data2=Data;
%类别2
for rows=1:size(Data,1)
    if Data(rows,3)~=3
        new_Data2(rows,3)=-1;
    else
        new_Data2(rows,3)=1;
    end
end

new_Data3=Data;
%类别3
for rows=1:size(Data,1)
    if Data(rows,3)~=5
        new_Data3(rows,3)=-1;
    else
        new_Data3(rows,3)=1;
    end
end
%使用SVM模型对数据集分三类，并绘制出分类结果的散点图和等高线图

%坐标数据
X1=new_Data1(:,1:2);
%标签数据
Y1=new_Data1(:,3);
%划分训练集和测试集，留40%为测试集
cvp = cvpartition(Data(:,3), 'Holdout', 0.40);
train_X1=X1(training(cvp),:);
train_Y1=Y1(training(cvp));
test_X1 = X1(test(cvp), :);
test_Y1 = Y1(test(cvp));
SVMModel1 =fitcsvm(train_X1, train_Y1,'BoxConstraint',10,'KernelFunction','rbf','KernelScale',2^0.5*2);%使用fitcsvm函数拟合训练集train_X1和train_Y1，其中train_X1为训练数据的特征向量，train_Y1为对应的类别标签。
pred_Y1 = predict(SVMModel1, test_X1);
figure;
gscatter(train_X1(:,1), train_X1(:,2), train_Y1, 'rbg','^*');
hold on;
d = 0.2;
[x1Grid1, x2Grid1] = meshgrid(min(train_X1(:,1)):d:max(train_X1(:,1)),min(train_X1(:,2)):d:max(train_X1(:,2)));
[label, scores] = predict(SVMModel1, [x1Grid1(:), x2Grid1(:)]); %用预先训练好的SVM模型（SVMModel3）对输入数据（x1Grid3, x2Grid3）进行预测，并返回预测的标签和置信度分数。
contour(x1Grid1, x2Grid1, reshape(scores(:,2), size(x1Grid1,1),size(x2Grid1,2)),[0 0],'r'); 

%坐标数据
X2=new_Data2(:,1:2);
%标签数据
Y2=new_Data2(:,3);
%划分训练集和测试集，留40%为测试集
cvp = cvpartition(Data(:,3), 'Holdout', 0.40);
train_X2=X2(training(cvp),:);
train_Y2=Y2(training(cvp));
test_X2 = X2(test(cvp), :);
test_Y2 = Y2(test(cvp));
SVMModel2 =fitcsvm(train_X2, train_Y2,'BoxConstraint',10,'KernelFunction','rbf','KernelScale',2^0.5*2);
pred_Y2 = predict(SVMModel2, test_X2);
figure;
gscatter(train_X2(:,1), train_X2(:,2), train_Y2, 'rbg','^*');
hold on;
d = 0.2;
[x1Grid2, x2Grid2] = meshgrid(min(train_X2(:,1)):d:max(train_X2(:,1)),min(train_X2(:,2)):d:max(train_X2(:,2)));
[label, scores] = predict(SVMModel2, [x1Grid2(:), x2Grid2(:)]); 
contour(x1Grid2, x2Grid2, reshape(scores(:,2), size(x1Grid2,1),size(x2Grid2,2)),[0 0],'b'); 

%坐标数据
X3=new_Data3(:,1:2);
%标签数据
Y3=new_Data3(:,3);
%划分训练集和测试集，留40%为测试集
cvp = cvpartition(Data(:,3), 'Holdout', 0.40);
train_X3=X3(training(cvp),:);
train_Y3=Y3(training(cvp));
test_X3 = X3(test(cvp), :);
test_Y3 = Y3(test(cvp));
SVMModel3 =fitcsvm(train_X3, train_Y3,'BoxConstraint',10,'KernelFunction','rbf','KernelScale',2^0.5*2);
pred_Y3 = predict(SVMModel3, test_X3);
d = 0.2;
[x1Grid3, x2Grid3] = meshgrid(min(train_X3(:,1)):d:max(train_X3(:,1)),min(train_X3(:,2)):d:max(train_X3(:,2)));
[label, scores] = predict(SVMModel3, [x1Grid3(:), x2Grid3(:)]); 
contour(x1Grid3, x2Grid3, reshape(scores(:,2), size(x1Grid3,1),size(x2Grid3,2)),[0 0],'g'); 

X=Data(:,1:3);
Y=Data(:,3);
old_labels = [1 2 3 4 5];
new_labels = [2 2 3 2 1];
for i = 1:length(old_labels)
    Y(Y == old_labels(i)) = new_labels(i);
end
cvp = cvpartition(Data(:,3), 'Holdout', 0.40);%创建了一个交叉验证分区对象，将Data矩阵的第三列作为分区依据，采用40%的数据作为测试集，其余数据作为训练集。
train_X=X(training(cvp),:);
train_Y=Y(training(cvp));
test_X = X(test(cvp), :);
test_Y = Y(test(cvp));
SVMModel =fitcecoc(train_X, train_Y);%使用训练集的输入特征和标签训练一个多类别的SVM模型
pred_Y = predict(SVMModel, test_X);%使用训练好的SVM模型对测试集的输入特征进行预测，得到预测的标签
fig2=figure;
plotconfusion(categorical(pred_Y),categorical(test_Y));%绘制预测结果和实际结果之间的混淆矩阵
title('混淆矩阵');
