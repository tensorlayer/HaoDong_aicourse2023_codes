# 第三课作业代码，以U-Net图像分割为例
import os
os.environ['TL_BACKEND'] = 'tensorflow' # 设置后端为TensorFlow

import matplotlib.pyplot as plt
import numpy as np

from tensorlayerx.dataflow import DataLoader # 导入数据加载器
from tlxzoo.datasets.circles import CirclesDataset # 导入数据集
from tlxzoo.module.unet import UnetTransform, crop_image_and_label_to_shape # 导入数据预处理
from tlxzoo.vision.image_segmentation import ImageSegmentation # 导入图像分割模型

if __name__ == '__main__':
    transform = UnetTransform() # 数据预处理
    test_dataset = CirclesDataset(100, transform=transform) # 加载数据集
    test_dataloader = DataLoader(test_dataset, batch_size=2) # 加载数据加载器

    model = ImageSegmentation(backbone="unet") # 加载模型,ImageSegmentation是多种图像分割模型的集合，backbone参数指定了使用哪种模型
    model.load_weights("./unet_model.npz") # 加载模型参数

    crop = crop_image_and_label_to_shape(transform.label_size)

    for batch_image, batch_label in test_dataloader:
        prediction = model.predict(batch_image)
        fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
        for i, (image, label) in enumerate(zip(batch_image, batch_label)):
            image, label = crop(image, label)
            ax[i][0].matshow(image[..., -1])
            ax[i][0].set_title('Original Image')
            ax[i][0].axis('off')
            ax[i][1].matshow(np.argmax(label, axis=-1), cmap=plt.cm.gray)
            ax[i][1].set_title('Original Mask')
            ax[i][1].axis('off')
            ax[i][2].matshow(np.argmax(prediction[i, ...].numpy(), axis=-1), cmap=plt.cm.gray)
            ax[i][2].set_title('Predicted Mask')
            ax[i][2].axis('off')
        plt.savefig("./circle.png")
        break