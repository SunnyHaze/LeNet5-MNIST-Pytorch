import struct
import matplotlib.pyplot as plt
import numpy as np
import requests , gzip
# 需要下载的文件名
fileNames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
    ]
# 用于简便直观理解文件名的字典
rawDataName = {
    "trainX" : "train-images-idx3-ubyte",
    "trainY" : "train-labels-idx1-ubyte",
    "testX" : "t10k-images-idx3-ubyte",
    "testY" : "t10k-labels-idx1-ubyte"
    }
# 用于下载文件的函数
def downLoadAFile(filename:str):
    baseUrl = 'http://yann.lecun.com/exdb/mnist/'
    tmpUrl = baseUrl + filename
    print('下载 {} 文件中...'.format(filename))
    file = requests.get(url=tmpUrl)
    with open(filename, 'wb+') as f:
        f.write(file.content)
    print('下载完成！')
    cutFilename = filename.replace('.gz','')
    print('解压gzip文件 {} 中...'.format(filename))
    gFile = gzip.GzipFile(filename)
    with open(cutFilename, 'wb+') as f:
        f.write(gFile.read())
    print('解压完成')
    print('===数据集下载完成===')
    
# =================以下是构建实际应用的数据集部分================
# 原始的数据是以“一个字节存储一个数值”的形式存储在数据集中的，所以是高度压缩的
# 而计算机中用来运算的浮点数，则需要达到32位（4字节），这也是显卡大多最支持的数据类型
# 如果不提前将数据转储位4字节格式，则会在读取数据时不断的由CPU进行运算，转换1字节的数据为4字节
# 这会重复浪费大量的运算能力，带来的结果就是CPU满载100%，但GPU几乎长期只有0%
# 而提前将数据解压好后，经过系统优化可以直接将整块的数据存入显存，大幅提高运算速度

# 虽然数据集的大小变大了不少，但是运算时间大幅降低！也就是下面定义的类的主要功能：
# 【读取高度压缩的字节码文件，并转化为GPU喜闻乐见的形式保存】
# （具体的字节码如何组织的，可以参考http://yann.lecun.com/exdb/mnist/ 网页页最下面的说明）
class imgReader:
    def __init__(self,PATH) -> None:
        self.path = PATH
        with open(self.path, 'rb') as f:
            self.buff = f.read()
             # 按字节拆出“元素”个数 具体字节码组织形式参考MNIST官网
            self.size = struct.unpack(">i",self.buff[4:8])[0] # 按照4字节无符号整数拆分字节，教程：https://www.liaoxuefeng.com/wiki/1016959663602400/1017685387246080
            # 象征性的拆一下横竖
            self.numberOfRows = struct.unpack(">i",self.buff[8:12]) 
            self.numberOfCols = struct.unpack(">i",self.buff[12:16])
    # 最好别用，贼慢，事实上下面的程序也没有使用此函数
    def returnWholeArray(self) -> list:
        data = {}
        data["imgs"] = []
        data["size"] = self.size
        for i in range(data["size"]):           
            tmpMatrix = struct.unpack_from('>784B',self.buff[16:],i * 784) # "magic number" 单纯的28x28的图像，784字节
            pic = np.array(tmpMatrix)
            pic = pic.reshape(28,28)
            data["imgs"].append(pic)
        return data
    # 重载[]运算符
    def __getitem__(self,index):
        if  type(index) == int:
            assert index >= 0 and index < self.size ,"Index out of range! Should less than {}.".format(self.size)
            offset = index * 784
            tmpMatrix = struct.unpack_from('>784B',self.buff[16:],offset)
            tmpMatrix = np.array(tmpMatrix).reshape(28,28)
            return np.array(tmpMatrix)
        if type(index) == slice:
            stop = self.size-1 if index.stop == None else index.stop
            start = 0 if index.start == None else index.start 
            length = stop - start
            data = []
            for i in range(start,stop):
                data.append(self.__getitem__(i))
            return data
    # 重载len()运算符
    def __len__(self):
        return self.size

# 用于读取标签的函数
class labelReader:
    def __init__(self,PATH) -> None:
        self.path = PATH
        with open(self.path, 'rb') as f:
            self.buff = f.read()
            self.size = struct.unpack(">i",self.buff[4:8])[0]
        # 最好别用，贼慢
    def returnWholeArray(self) -> list:
        data = {}
        data["imgs"] = []
        sizeByte = self.buff[4:8] # 按字节拆出“元素”个数 具体字节码组织形式参考MNIST官网
        data["size"] = struct.unpack(">i",sizeByte)[0] # 按照4字节无符号整数拆分字节，教程：https://www.liaoxuefeng.com/wiki/1016959663602400/1017685387246080
        # 象征性的拆一下横竖
        for i in range(data["size"]):           
            tmpMatrix = struct.unpack_from('>B',self.buff[8:],i) # "magic number" 单纯的28x28的图像，784字节
            pic = np.array(tmpMatrix)
            pic = pic.reshape(28,28)
            data["imgs"].append(pic)
        return data
    # 重载[]运算符
    def __getitem__(self,index):
        if  type(index) == int:
            assert index >= 0 and index < self.size ,"Index {} out of range! Should less than {}.".format(index,self.size)
            offset = index 
            label= struct.unpack_from('>B',self.buff,offset + 8)
            return label[0]
        if type(index) == slice:
            stop = self.size-1 if index.stop == None else index.stop
            start = 0 if index.start == None else index.start 
            length = stop - start
            label = struct.unpack_from(">" + str(length) + "B", self.buff,8 + start)
            return label
    # 重载len()运算符
    def __len__(self):
        return self.size
# 将一个Reader中的所有数据，1字节解码为4字节后，转储到fileName对应的文件中
def saveAsNpy(reader,fileName):
    with open(fileName,'wb') as f:
        data = []
        for i in range(len(reader)):
            if i % 100 == 0:
                print('{:.2f}%'.format(i/len(reader) * 100))
            data.append(reader[i])
        data = np.array(data)
        print('Done!')
        np.save(f,data)
# 主函数
if __name__ == "__main__":  
    # 从网站http://yann.lecun.com/exdb/mnist/自动下载MNIST数据集并解压
    for i in fileNames:
        downLoadAFile(i)  
    # 组织好Reader，用于读取格式化的字节码  
    trainXreader = imgReader(rawDataName["trainX"])
    testXreader = imgReader(rawDataName["testX"])
    testYreader = labelReader(rawDataName["testY"])
    trainYreader = labelReader(rawDataName["trainY"])
    
    print('===正在转换训练集图片，总共{}个...==='.format(len(trainXreader)))
    saveAsNpy(trainXreader,'TrainImg.npy')
    print('===正在转换测试集图片，总共{}个...'.format(len(testXreader)))
    saveAsNpy(testXreader,'TestImg.npy')
    print('===正在转换测试集标签，总共{}个...'.format(len(testYreader)))
    saveAsNpy(testYreader,"TestLabel.npy")
    print('===正在转换训练集标签，总共{}个...'.format(len(trainYreader)))
    saveAsNpy(trainYreader,"TrainLabel.npy")

    print('=====恭喜您！全部完成，可以开始体验训练模型了！======')

