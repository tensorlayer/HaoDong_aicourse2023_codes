# 第一课自定义激活函数的作业代码
import os
os.environ['TL_BACKEND'] = 'torch'  # 设置后端为pytorch

import tensorlayerx as tlx  # 导入tensorlayerx编程框架库


class ActSigmoid(tlx.nn.Module):
    '''
    Sigmoid 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = 1 / (1 + exp(-x))
        '''
        return 1 / (1 + tlx.exp(-x))


class ActTanh(tlx.nn.Module):
    '''
    Tanh 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        '''
        return tlx.tanh(x)  # 直接调用tlx.tanh()函数


class ActLeakyReLU(tlx.nn.Module):
    '''
    LeakyReLU 激活函数
    '''

    def forward(self, x):
        '''
        前向传播, 数学公式：
        y = max(0.01*x, x)
        '''
        return tlx.maximum(x, 0.01*x)


if __name__ == "__main__":
    # Test
    # 定义输入向量
    x = tlx.convert_to_tensor([[1., 2., 3.]])
    w = tlx.convert_to_tensor([[-0.5], [0.2], [0.1]])
    b1 = tlx.convert_to_tensor(0.5)

    # 矩阵乘法
    z1 = tlx.matmul(x, w)+b1
    print("Z:\n", z1, "\nShape", z1.shape)

    # 测试激活函数
    # Sigmoid function
    sigmoid_tlx = tlx.nn.activation.Sigmoid()  # TLX内置的Sigmoid激活函数对象
    a1 = sigmoid_tlx(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result tlx sigmoid:", a1)  # 使用TLX内置函数的结果

    sigmoid_act = ActSigmoid() # 自定义的Sigmoid激活函数对象
    a2 = sigmoid_act(z1) # 调用对象的__call__方法，执行前向传播
    print("Result act sigmoid:", a2) # 使用自定义的激活函数的结果

    # Tanh
    tanh_tlx = tlx.nn.activation.Tanh() # TLX内置的Tanh激活函数对象
    a3 = tanh_tlx(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result tlx Tanh:", a3) # 使用TLX内置函数的结果

    tanh_act = ActTanh() # 自定义的Tanh激活函数对象
    a4 = tanh_act(z1) # 调用对象的__call__方法，执行前向传播
    print("Result act Tanh:", a4) # 使用自定义的激活函数的结果

    # Leaky ReLU
    leakyrelu_tlx = tlx.nn.activation.LeakyReLU() # TLX内置的LeakyReLU激活函数对象
    a5 = leakyrelu_tlx(z1) # 调用对象的__call__方法，执行前向传播
    print("Result tlx LeakyReLU:", a5) # 使用TLX内置函数的结果

    leakyrelu_act = ActLeakyReLU() # 自定义的LeakyReLU激活函数对象
    a6 = leakyrelu_act(z1)  # 调用对象的__call__方法，执行前向传播
    print("Result act LeakyReLU:", a6) # 使用自定义的激活函数的结果
