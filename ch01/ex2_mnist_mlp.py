# 第一课作业使用TensorLayerX训练MNIST数据集的MLP模型
import os 
os.environ['TL_BACKEND'] = 'torch' #设置后端为pytorch

# 导入tensorlayerx编程框架库
import tensorlayerx as tlx
from tensorlayerx.nn import Linear

# 导入数据处理相关库
from tensorlayerx.dataflow import Dataset, DataLoader


#使用TLX的文件API加载MNIST数据集的数据
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(
    shape=(-1, 784))

print("Train data shape:{}".format(X_train.shape))

# 定义MNIST数据集类
class MNISTDataset(Dataset):#继承Dataset类

    def __init__(self, data=X_train, label=y_train):
        '''
        Args:
            data: numpy array, shape=(N, 784)
            label: numpy array, shape=(N, 10)
        '''
        self.data = data
        self.label = label

    def __getitem__(self, index):
        '''
        根据索引获取数据,返回数据和标签,一个tuple
        '''
        data = self.data[index].astype('float32') #转换数据类型, 神经网络一般使用float32作为输入的数据类型
        label = self.label[index].astype('int64') #转换数据类型, 分类任务神经网络一般使用int64作为标签的数据类型
        return data, label

    def __len__(self):
        '''
        返回数据集的样本数量
        '''
        return len(self.data)


# 构建神经网络模型

class CustomModel(tlx.nn.Module):
    '''
    这是一个自定义的MLP模型
    '''
    def __init__(self):
        '''
        这个网络包含三层全连接层
        第一层是输入层, 输入数据的维度是784, 输出数据的维度是800
        第二层是隐藏层, 输入数据的维度是800, 输出数据的维度是800
        第三层是输出层, 输入数据的维度是800, 输出数据的维度是10, 代表10个类别
        '''
        super(CustomModel, self).__init__() #调用父类的构造函数

        #使用Linear层构建全连接层
        self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.linear3 = Linear(out_features=10, in_features=800)

    def forward(self, x):
        '''
        定义网络的前向传播过程
        '''
        a = self.linear1(x)
        a = self.linear2(a)
        out = self.linear3(a)

        return out

# 实例化模型
MLP = CustomModel()

# 定义训练参数
n_epoch = 50 #训练50个epoch
batch_size = 128 #每个batch包含128个样本
print_freq = 1 #每训练1个epoch打印一次训练信息

#设置损失函数和评价指标
loss_fn = tlx.losses.softmax_cross_entropy_with_logits # 分类任务，使用交叉熵损失函数
metric = tlx.metrics.Accuracy() # 分类任务，使用准确率作为评价指标

# 设置优化器
optimizer = tlx.optimizers.Adam(lr=0.01) # 使用Adam优化器

# 构建数据迭代器

# 训练集
train_dataset = MNISTDataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 测试集
test_dataset = MNISTDataset(data=X_test, label=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 使用TensorlayerX高级API进行训练
# 封装训练过程
net_with_train = tlx.model.Model(
    network=MLP, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
print("Start training\n")

# 开始训练
net_with_train.train(n_epoch=n_epoch, train_dataset=train_loader,
                    # test_dataset=test_loader,
                     print_freq=print_freq, print_train_batch=False)
