# 第三课卷积神经网络配套代码
import os
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx

# Convolution on 1D vector 一维卷积

x = tlx.convert_to_tensor([[[1.], [3.], [3.], [0.], [1.], [2.]]]) # 输入向量，一维向量的维度为[1, 6, 1]
w = tlx.convert_to_tensor([[[2., 0., 1.]]]) # 卷积核，一维卷积核的维度为[1, 1, 3]

conv1d = tlx.ops.Conv1D(stride=1, padding='VALID', dilations=1) # 一维卷积层
out = conv1d(x, w) # 卷积运算

print("Input vector: {}".format(x.flatten().tolist()))
print("Convolution filter: {}".format(w.flatten().tolist()))
print("Result: {}".format(out.flatten().tolist()))

# VGG16
from vgg import VGG16
image=tlx.convert_to_tensor(tlx.ones(shape=[1, 224, 224, 3]))

# 完整的VGG16模型，包含最后的全连接层，直接用于分类
model_entire = VGG16(end_with="outputs",name="entire")

y_out=model_entire(image)

# VGG16模型，只包含卷积层和池化层，用于特征提取
model_feature = VGG16(end_with="pool5",name="feature")
y_feature=model_feature(image)