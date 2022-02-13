from imports.ParametersManager import *
from matplotlib import pyplot as plt
parManager = ParametersManager('cuda')
parManager.loadFromFile('model.pt')
# 在终端输出最终的两个数据集准确率
parManager.show()

# 绘制迄今为止的训练准确率图
plt.figure(figsize=(10,7))
plt.plot(range(parManager.EpochDone),parManager.TrainACC,marker='*' ,color='r',label='Train')
plt.plot(range(parManager.EpochDone),parManager.TestACC,marker='*' ,color='b',label='Test')

plt.xlabel('Epochs')
plt.ylabel('ACC')
plt.legend()
plt.title("LeNet-5 on MNIST")

plt.show()