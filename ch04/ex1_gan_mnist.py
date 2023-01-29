#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ['TL_BACKEND'] = 'torch'

import time
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn import Module, Linear
from tensorlayerx.dataflow import Dataset
from tensorlayerx.model import TrainOneStep

import torch
# device = torch.device('mlu:0' if torch.mlu.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 784))


class MNISTDataset(Dataset):

    def __init__(self, data=X_train, label=y_train):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)


batch_size = 128
train_dataset = MNISTDataset(data=X_train, label=y_train)
train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class generator(Module):

    def __init__(self):
        super(generator, self).__init__()
        self.g_fc1 = Linear(out_features=256, in_features=100, act=tlx.nn.ReLU)
        self.g_fc2 = Linear(out_features=256, in_features=256, act=tlx.nn.ReLU)
        self.g_fc3 = Linear(out_features=784, in_features=256, act=tlx.nn.Tanh)

    def forward(self, x):
        out = self.g_fc1(x)
        out = self.g_fc2(out)
        out = self.g_fc3(out)
        return out


class discriminator(Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.d_fc1 = Linear(out_features=256, in_features=784, act=tlx.LeakyReLU)
        self.d_fc2 = Linear(out_features=256, in_features=256, act=tlx.LeakyReLU)
        self.d_fc3 = Linear(out_features=1, in_features=256, act=tlx.Sigmoid)

    def forward(self, x):
        out = self.d_fc1(x)
        out = self.d_fc2(out)
        out = self.d_fc3(out)
        return out


G = generator()
D = discriminator()

class WithLossG(Module):

    def __init__(self, G, D, loss_fn):
        super(WithLossG, self).__init__()
        self.g_net = G
        self.d_net = D
        self.loss_fn = loss_fn

    def forward(self, g_data, label):
        fake_image = self.g_net(g_data)
        logits_fake = self.d_net(fake_image)
        valid = tlx.convert_to_tensor(np.ones(logits_fake.shape), dtype=tlx.float32).to(logits_fake.device)
        loss = self.loss_fn(logits_fake, valid)
        return loss


class WithLossD(Module):

    def __init__(self, G, D, loss_fn):
        super(WithLossD, self).__init__()
        self.g_net = G
        self.d_net = D
        self.loss_fn = loss_fn

    def forward(self, real_data, g_data):
        logits_real = self.d_net(real_data)
        fake_image = self.g_net(g_data)
        logits_fake = self.d_net(fake_image)

        valid = tlx.convert_to_tensor(np.ones(logits_real.shape), dtype=tlx.float32).to(logits_real.device)
        fake = tlx.convert_to_tensor(np.zeros(logits_fake.shape), dtype=tlx.float32).to(logits_real.device)

        loss = self.loss_fn(logits_real, valid) + self.loss_fn(logits_fake, fake)
        return loss


# loss_fn = tlx.losses.sigmoid_cross_entropy
# optimizer = tlx.optimizers.Momentum(learning_rate=5e-4, momentum=0.5)
loss_fn = tlx.losses.mean_squared_error
optimizer_g = tlx.optimizers.Adam(lr=3e-4, beta_1=0.5, beta_2=0.999)
optimizer_d = tlx.optimizers.Adam(lr=3e-4)

g_weights = G.trainable_weights
d_weights = D.trainable_weights
net_with_loss_G = WithLossG(G.to(device), D.to(device), loss_fn).to(device)
net_with_loss_D = WithLossD(G.to(device), D.to(device), loss_fn).to(device)
train_one_step_g = TrainOneStep(net_with_loss_G, optimizer_g, g_weights)
train_one_step_d = TrainOneStep(net_with_loss_D, optimizer_d, d_weights)
n_epoch = 50


def plot_fake_image(fake_image, num):
    fake_image = tlx.reshape(fake_image, shape=(num, 28, 28))
    fake_image = tlx.convert_to_numpy(fake_image)
    import matplotlib.pylab as plt
    for i in range(num):
        plt.subplot(int(np.sqrt(num)), int(np.sqrt(num)), i + 1)
        plt.imshow(fake_image[i])
    plt.show()


for epoch in range(n_epoch):
    d_loss, g_loss = 0.0, 0.0
    n_iter = 0
    start_time = time.time()
    for data, label in train_loader:
        noise = tlx.convert_to_tensor(np.random.random(size=(batch_size, 100)), dtype=tlx.float32)

        _loss_d = train_one_step_d(data.to(device), noise.to(device))
        _loss_g = train_one_step_g(noise.to(device), label.to(device))
        d_loss += _loss_d
        g_loss += _loss_g

        n_iter += 1
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   d loss: {}".format(d_loss / n_iter))
        print("   g loss:  {}".format(g_loss / n_iter))
    fake_image = G(tlx.convert_to_tensor(np.random.random(size=(36, 100)), dtype=tlx.float32).to(device))
    plot_fake_image(fake_image, 36)
