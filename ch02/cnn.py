import os
os.environ['TL_BACKEND'] = 'paddle'

import tensorlayerx as tlx

# Convolution on 1D vector

x = tlx.convert_to_tensor([[[1.], [3.], [3.], [0.], [1.], [2.]]])
w = tlx.convert_to_tensor([[[2., 0., 1.]]])
conv1d = tlx.ops.Conv1D(stride=1, padding='VALID', dilations=1)

out = conv1d(x, w)
print("Input vector: {}".format(x.flatten().tolist()))
print("Convolution filter: {}".format(w.flatten().tolist()))
print("Result: {}".format(out.flatten().tolist()))

# VGG16
from tlxzoo.module import VGG
image=tlx.convert_to_tensor(tlx.ones(shape=[1, 32, 32, 3]))

model_entire = VGG(layer_type="vgg16",end_with="fc1_relu", num_labels=10)

y=model_entire(image)

# model_feature = VGG(layer_type="vgg16",end_with="pool5",name="feature")
# y=model_feature(image)