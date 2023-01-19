#Train your first neural network
import os
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from tensorlayerx.nn import Linear

from tensorlayerx.dataflow import Dataset, DataLoader


#Load MNIST dataset
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(
    shape=(-1, 784))

print("Train data shape:{}".format(X_train.shape))

#Prepare Dataset
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


#Build model
class CustomModel(tlx.nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.linear3 = Linear(out_features=10, in_features=800)

    def forward(self, x):
        a = self.linear1(x)
        a = self.linear2(a)
        out = self.linear3(a)
        return out


MLP = CustomModel()

#Set training parameters
n_epoch = 50
batch_size = 128
print_freq = 1

#Set loss function and metric
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
metric = tlx.metrics.Accuracy()

#Optimizer
optimizer = tlx.optimizers.Adam(lr=0.01)

train_dataset = MNISTDataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNISTDataset(data=X_test, label=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Use TensorlayerX high-level training API
#Warp training process
net_with_train = tlx.model.Model(
    network=MLP, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
print("Start training\n")
net_with_train.train(n_epoch=n_epoch, train_dataset=train_loader,
                    test_dataset=test_loader,
                     print_freq=print_freq, print_train_batch=False)
