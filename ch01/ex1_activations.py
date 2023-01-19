import os
os.environ['TL_BACKEND'] = 'paddle'

import tensorlayerx as tlx

class ActSigmoid(tlx.nn.Module):
    def forward(self, x):
        return 1 / (1 + tlx.exp(-x))

class ActTanh(tlx.nn.Module):
    def forward(self, x):
        return tlx.tanh(x)


class ActLeakyReLU(tlx.nn.Module):
    def forward(self, x):
        return tlx.maximum(x, 0.01*x)

#Test
x = tlx.convert_to_tensor([[1., 2., 3.]])
w = tlx.convert_to_tensor([[-0.5], [0.2], [0.1]])
b1 = tlx.convert_to_tensor(0.5)
z1 = tlx.matmul(x, w)+b1
print("Z:\n", z1, "\nShape", z1.shape)

# Sigmoid function
sigmoid_tlx = tlx.nn.activation.Sigmoid()
a1 = sigmoid_tlx(z1)
print("Result tlx sigmoid:", a1)

sigmoid_act = ActSigmoid()
a2 = sigmoid_act(z1)
print("Result act sigmoid:", a2)

# Tanh
tanh_tlx = tlx.nn.activation.Tanh()
a3 = tanh_tlx(z1)
print("Result tlx Tanh:", a3)

tanh_act = ActTanh()
a4 = tanh_act(z1)
print("Result act Tanh:", a4)

# Leaky ReLU
leakyrelu_tlx = tlx.nn.activation.LeakyReLU()
a5 = leakyrelu_tlx(z1)
print("Result tlx LeakyReLU:", a5)

leakyrelu_act = ActLeakyReLU()
a6 = leakyrelu_act(z1)
print("Result act LeakyReLU:", a6)
