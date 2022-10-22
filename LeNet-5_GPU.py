import os
from matplotlib.pyplot import imshow
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from imports.ParametersManager import *

# 超参数
BatchSize = 10
LEARNINGRATE = 0.005
epochNums = 30
SaveModelEveryNEpoch = 2 # 每执行多少次保存一个模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 初始化数据转换器，通过索引访问
# trainXreader = imgReader(rawDataName["trainX"])
# testXreader = imgReader(rawDataName["testX"])
# trainYreader = labelReader(rawDataName["trainY"])
# testYreader = labelReader(rawDataName["testY"])

# 可以将数据线包装为Dataset，然后传入DataLoader中取样
class MyDataset(Dataset):
    def __init__(self,SetType) -> None:
        with open(SetType + 'img.npy','rb') as f:
            self.images =torch.tensor(np.load(f, allow_pickle=True), dtype=torch.float32)
        with open(SetType + 'Label.npy','rb') as f:
            tmp = np.load(f, allow_pickle=True)
            print(tmp)
        self.labels=[]
        for num in tmp:
             self.labels.append([1 if x == num else 0 for x in range(10)])
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    def __getitem__(self, index):
        return self.images.unsqueeze(1)[index], self.labels[index]
    def __len__(self):
        return len(self.labels)
    
# 定义网络结构
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),# 原题为三通道，此处转为单通道实现 # C1
            nn.ReLU(),
            nn.MaxPool2d(2,2), # S2
            nn.Conv2d(6,16,5), # C3  原始论文中C3与S2并不是全连接而是部分连接，这样能减少部分计算量。而现代CNN模型中，比如AlexNet，ResNet等，都采取全连接的方式了。我们的实现在这里做了一些简化。
            nn.ReLU(),
            nn.MaxPool2d(2,2) # S4
        )
        # 然后需要经过变形后，继续进行全连接
        self.layer2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), # C5
            nn.ReLU(),
            nn.Linear(120, 84),         # F6
            nn.ReLU(),
            nn.Linear(84,10), # Output 文章中使用高斯连接，现在方便起见仍然使用全连接
        )
    def forward(self,x):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = x.view(-1,16 * 5 * 5) # 重新构建向量形状，准备全连接
        x = self.layer2(x) # 执行全连接部分
        return x

# 定义准确率函数
def accuracy(output , label):
    rightNum = torch.sum(torch.max(label,1)[1].eq(torch.max(output,1)[1]))
    return rightNum / len(label)
        
if __name__ == "__main__":    
    # 模型实例化        
    model = LeNet_5()
    # # 如果有“半成品”则导入参数
    parManager = ParametersManager(device)
    if os.path.exists("./model.pt"):
        parManager.loadFromFile('./model.pt')
        parManager.setModelParameters(model)
    else:
        print('===No pre-trained model found!===')

    model.cuda()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNINGRATE, momentum=0.9)
    
    # 构建训练集
    TrainDataset = MyDataset('Train')
    # 构建测试集
    TestDataset = MyDataset('Test')
    # 构建训练集读取器
    TrainLoader = DataLoader(TrainDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TrainDataset))))
    # 构建测试集读取器：
    TestLoader = DataLoader(TestDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TestDataset))))
    # 
    print('len(TrainLoader):{}'.format(len(TrainLoader)))
    
    # # 检查分割是否正确的函数，分为两行，以行为顺序排列和输出结果一一对应
    # def testLoader():
    #     inputs, classes = next(iter(TrainLoader))
    #     print(inputs.shape)
    #     print(classes.shape)
    #     print(classes) # 查看标签
    #     for i in range(len(inputs)):
    #         plt.subplot(2,5,i+1)
    #         plt.imshow(inputs[i][0],cmap="gray")
    #     plt.show()
        
    # testLoader()

    TrainACC = []
    TestACC = []
    GlobalLoss = []
    for epoch in range(epochNums):
        print("===开始本轮的Epoch {} == 总计是Epoch {}===".format(epoch, parManager.EpochDone))
        
        # 收集训练参数
        epochAccuracy = []
        epochLoss = []
        #=============实际训练流程=================
        for batch_id, (inputs,label) in enumerate(TrainLoader):
            # 先初始化梯度0
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = criterion(output,label.cuda())
            loss.backward()
            optimizer.step()
            epochAccuracy.append(accuracy(output,label.cuda()).cpu())
            epochLoss.append(loss.item()) # 需要获取数值来转换
            if batch_id % (len(TrainLoader) / 20) == 0: 
                print("    当前运行到[{}/{}], 目前Epoch准确率为：{:.2f}%，Loss：{:.6f}".format(batch_id,len(TrainLoader), np.mean(epochAccuracy) * 100, loss))
        #==============本轮训练结束==============
        # 收集训练集准确率
        TrainACC.append(np.mean(epochAccuracy)) 
        GlobalLoss.append(np.mean(epochLoss))
        # ==========进行一次验证集测试============
        localTestACC = []
        model.eval() # 进入评估模式，节约开销
        for inputs, label in TestLoader:
            torch.no_grad() # 上下文管理器，此部分内不会追踪梯度
            output = model(inputs.cuda())
            localTestACC.append(accuracy(output,label.cuda()).cpu())
        # ==========验证集测试结束================
        # 收集验证集准确率
        TestACC.append(np.mean(localTestACC))
        print("当前Epoch结束，训练集准确率为：{:3f}%，测试集准确率为：{:3f}%".format(TrainACC[-1] * 100, TestACC[-1] * 100))
        # 暂存结果到参数管理器
        parManager.oneEpochDone(LEARNINGRATE,TrainACC[-1],TestACC[-1],GlobalLoss[-1])
        # 周期性保存结果到文件
        if epoch == epochNums - 1 or epoch % SaveModelEveryNEpoch == 0:
            parManager.loadModelParameters(model)
            parManager.saveToFile('./model.pt')
            
    # 查看此次训练之后结果
    parManager.show()
    # 绘图
    plt.figure(figsize=(10,7))
    plt.plot(range(parManager.EpochDone),parManager.TrainACC,marker='*' ,color='r',label='Train')
    plt.plot(range(parManager.EpochDone),parManager.TestACC,marker='*' ,color='b',label='Test')

    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.title("LeNet-5 on MNIST")

    plt.savefig('Train.jpg')
    plt.show()