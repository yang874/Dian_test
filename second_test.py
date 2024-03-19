import torch
import torch.nn as nn  #导入 PyTorch 中的神经网络模块（nn）
import torch.optim as optim   #提供了一个通用的优化器接口
from torch.utils.data import DataLoader  #提供了一个 DataLoader 类，可以用来加载和迭代数据集
from torchvision import datasets  #提供了一系列常见的图像数据集，如 MNIST
from torchvision import transforms  #供了一系列常用的图像变换操作，如缩放、裁剪、旋转、归一化等

# global definitions
BATCH_SIZE = 100  #单次传递给程序用以训练的数据（样本）个数
MNIST_PATH = r"C:\Users\yang\Desktop\fashion-mnist"


# 把字节数据转化成Tensor，并按照一定标准归一化数据
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# 准备训练集数据
# training dataset
train_dataset = datasets.MNIST(root=MNIST_PATH,
                               train=True,  #代表我们读入的数据作为训练集（如果为true则从training.pt创建数据集，否则从test.pt创建数据集）
                               download=True,  #download=True是当根目录（root）下没有数据集时，自动下载
                               transform=transform)  #读入我们自己定义的数据预处理操作
# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE)

#准备测试集数据
# test dataset
test_dataset = datasets.MNIST(root=MNIST_PATH,
                              train=False,
                              download=True,
                              transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=BATCH_SIZE)


# 定义模型
#简单的三层全连接神经网络
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        #将input和之前的网络中的隐藏层参数合并。
        combined = torch.cat((input, hidden), 1) 

        hidden = self.i2h(combined) #计算隐藏层参数
        output = self.i2o(combined) #计算网络输出的结果

        return output, hidden

    def init_hidden(self, batch_size):
        #初始化隐藏层参数hidden
        return torch.zeros(batch_size, self.hidden_size)


#定义学习率，训练次数，损失函数，优化器
    
learning_rate = 1e-2
epoches = 20
criterion = nn.CrossEntropyLoss()
model = RNN(28*28,128,10)
optimizer = optim.SGD(model.parameters(),lr=learning_rate)
num_classes = 10  # 设定类别数量

#定义指标数组
Acc = []
Pre = []
Rec = []
F1 = []

#模型进行训练

for epoch in range(epoches):
    train_loss = 0
    train_acc = 0

    # 初始化TP, FP, FN计数器
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for img,label in train_loader:
        hidden = model.init_hidden(img.shape[0])
        img = torch.Tensor(img.view(img.size(0),-1))
        label = torch.Tensor(label).long() # 模型期望接收 LongTensor 类型的标签
        output, hidden = model(img, hidden)  ## 传入隐藏状态 hidden
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _,pred = output.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

        for i in range(len(label)):
                    if label[i] == pred[i]:
                        tp[label[i]] += 1
                    else:
                        fp[pred[i]] += 1
                        fn[label[i]] += 1

        precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(num_classes)]
        recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(num_classes)]
        f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i] ) >0 else 0 for i in range(num_classes)]

        # 计算每个类别的平均精确度、召回率和F1分数
        avg_precision = sum(precision) / num_classes
        avg_recall = sum(recall) / num_classes
        avg_f1_score = sum(f1_score) / num_classes

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Pre: {:.6f},Train Rec: {:.6f}, Train F1_Score: {:.6f}'.\
            format(epoch+1, train_loss/len(train_loader), train_acc/len(train_loader), avg_precision, avg_recall, avg_f1_score))

    #数组填充
    Acc.append(train_acc/len(train_loader))
    Pre.append(avg_precision)
    Rec.append(avg_recall)
    F1.append(avg_f1_score)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,8),dpi=150) # 设置图片大小
plt.title('指标')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('epoch')  # x轴标题
plt.ylabel('%')  # y轴标题
x = range(1 , epoches + 1)  # x = epoch

#plt.plot(x, loss, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, Acc, marker='o', markersize=3)
plt.plot(x, Pre, marker='o', markersize=3)
plt.plot(x, Rec, marker='o', markersize=3)
plt.plot(x, F1, marker='o', markersize=3)
    
plt.legend(['准确率', '精确率', '召回率', 'F1分数'])  # 设置折线名称

plt.show()  # 显示折线图

plt.savefig("C:/Users/yang/Desktop/dian_test/index_image.png")        #将图片保存到本地


