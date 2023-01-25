# 第一课课上配套代码
import os
os.environ['TL_BACKEND'] = 'torch'

# 导入TensorLayerX库
import tensorlayerx as tlx
from tensorlayerx.nn import Linear
from tensorlayerx.nn import Sequential


tlx.set_seed(99999)  # set random set

# Single neuro
# 单个神经元

# matrix multiplication 矩阵乘法
x = tlx.convert_to_tensor([[1., 2., 3.]]) # 1x3
w = tlx.convert_to_tensor([[-0.5], [0.2], [0.1]]) # 3x1

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b1 = tlx.convert_to_tensor(0.5)

# 计算有偏置的矩阵乘法
z1 = tlx.matmul(x, w)+b1

print("Z:\n", z1, "\nShape", z1.shape)

# Two outputs 两个输出

x = tlx.convert_to_tensor([[1., 2., 3.]]) # 1x3

w = tlx.convert_to_tensor([[-0.5, -0.3],
                           [0.2, 0.4],
                           [0.1, 0.15]]) # 3x2

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b2 = tlx.convert_to_tensor([0.5, 0.4])

z2 = tlx.matmul(x, w)+b2

print("Z:\n", z2, "\nShape", z2.shape)


# Activation function 激活函数
# Reference: https://en.wikipedia.org/wiki/Activation_function
# Sigmoid function
sigmoid_tlx = tlx.nn.activation.Sigmoid()
a1 = sigmoid_tlx(z1)
print("Result tlx sigmoid:", a1)

# define your own activation function


class ActSigmoid(tlx.nn.Module):
    def forward(self, x):
        return 1 / (1 + tlx.exp(-x))


sigmoid_act = ActSigmoid()
a1_m = sigmoid_act(z1)

print("Result your own sigmoid:", a1_m)

# Softmax
a2 = tlx.softmax(z2, axis=-1)
print("Result tlx softmax:", a2)


class ActSoftmax(tlx.nn.Module):
    def forward(self, x, axis=-1):
        e = tlx.exp(x - tlx.reduce_max(x, axis=axis, keepdims=True))
        s = tlx.reduce_sum(e, axis=axis, keepdims=True)
        return e / s


softmax_act = ActSoftmax()
a2_m = softmax_act(z2, axis=-1)
print("Result your own softmax:", a2_m)


# MLP 多层感知机
# 构建序列式模型，自动计算输入输出维度
layer_list = []
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear1'))
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear2'))
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear3'))
layer_list.append(Linear(out_features=2, act=tlx.Softmax,
                  in_features=3, name='linear4'))
MLP = Sequential(layer_list)

out = MLP(x)
print("Neural network output: ", out.shape)

# Get parameters of the model
# 获取模型参数
all_weights = MLP.all_weights
some_weights = MLP.all_weights[1:3]

linear1_weights = MLP.all_weights[0] # MLP[0].all_weights
# Loss functions 损失函数
# Reference: https://en.wikipedia.org/wiki/Loss_function

# Mean absolute error 平均绝对误差
def mae(output, target):
    return tlx.reduce_mean(tlx.abs(output-target), axis=-1)

# Mean squared error 平均平方误差
def mse(output, target):
    return tlx.reduce_mean(tlx.square(output-target), axis=-1)


y1 = tlx.convert_to_tensor([1., 3., 5., 7.])
y2 = tlx.convert_to_tensor([2., 4., 6., 8.])

# 计算MAE和MSE
l_mae = mae(y1, y2)
l_mse = mse(y1, y2)

print("Loss MAE: {} \nLoss MSE: {}".format(l_mae, l_mse))

l_mse_tlx = tlx.losses.mean_squared_error(y1, y2)
print("Loss MSE by TLX: {}".format(l_mse_tlx.numpy()))

# 二分类标签
target_binary = tlx.convert_to_tensor([[1., 0.]])

print("Target value:{}".format(target_binary.numpy()))
print("Neural network output:{}".format(out.detach().numpy()))

# Binary cross entropy 二分类交叉熵
l_bce = tlx.losses.binary_cross_entropy(out, target_binary)
print("Loss binary cross entropy:{}".format(l_bce.detach().numpy()))


# Error Back-Propagation 误差梯度反向传播
# 使用自动求导机制
x = tlx.Variable(tlx.convert_to_tensor(1.))
x.stop_gradient = False
w = tlx.Variable(tlx.convert_to_tensor(0.5))
w.stop_gradient = False

# 前向传播
t = x+w
z = t**2

# 反向传播
z.backward()

print("Gradient of w is: {}".format(w.grad.numpy()))

# BP for the network
l_bce.backward()

print("Gradient of the network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.grad.numpy()))

#Optimization 优化
opt = tlx.optimizers.SGD(lr=0.01, momentum=0.9) # 随机梯度下降优化器

x_in = tlx.convert_to_tensor([[1., 2., 3.]])
target = tlx.convert_to_tensor([[1., 0.]])

print("Before optimization, network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.detach().numpy()))

out = MLP(x_in) # 前向传播
l_bce = tlx.losses.binary_cross_entropy(out, target) # 计算损失

grads = opt.gradient(l_bce, MLP.trainable_weights) # 计算梯度
opt.apply_gradients(zip(grads, MLP.trainable_weights)) # 更新参数
print("After optimization, network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.detach().numpy()))


