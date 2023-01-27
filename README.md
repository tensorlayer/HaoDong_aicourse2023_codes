# 《深度学习入门与实践》配套代码
## 课程简介
* 授课教师：董豪 助理教授、研究员、博士生导师
* 所属机构：北京大学前沿计算研究中心
* 开课时间：2023年1月   
* “深度学习入门与实践”课程的主要目标，是希望学习者通过该课程初步掌握深度学习的理论知识、常用算法原理并且具备一定的应用开发能力。本课程特色一方面是兼顾理论和实践，既讲解了深度学习的底层算法原理，又介绍了常用的模型算法，同时还配套有开发框架的代码实践。
* 课程主要内容包括神经网络基础、卷积神经网络、对抗生成网络（GAN）、循环神经网络等常用算法原理，包括张量、激活函数、反向传播、优化算法等基础概念，并且结合编程框架进行代码开发实践。
* 课程配套代码使用[TensorLayerX框架](https://github.com/tensorlayer/TensorLayerX)开发，安装请参考[官方文档](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)。
  
## 课程目录
* [ch01 神经网络基础](ch01)  
  * 内容摘要：神经元、激活函数、全连接层、多层感知机、反向传播、优化器，MNIST 手写识别
  * 代码运行：
    * [课上演示代码](ch01/examples.py)：  
        ```
        python ch01/examples.py
        ```
    * 作业代码：  
        * [自定义激活函数](ch01/ex1_activations.py):
        ```
        python ch01/ex1_activations.py
        ```
        * [多层感知机MNIST 手写识别](ch01/ex2_mnist_mlp.py):
        ```
        python ch01/ex2_mnist_mlp.py
        ```
        * [numpy实现反向传播](ch01/ex3_bp.py):
        ```
        python ch01/ex3_bp.py
        ```

* [ch02 卷积神经网络基础](ch02)
  * 内容摘要：卷积层、池化层、特征提取与感受野、CIFAR10 图像分类、常用 backbone 算法
  * 代码运行：
    * 课上演示代码：
        ```
        python ch02/cnn.py
        ```
    * 作业代码：
      * [CIFAR10 图像分类](ch02/ex_cifar10_cnn.py)：
        ```
        python ch02/ex1_cifar10.py
        ```

* [ch03 计算机视觉应用](ch02)
  * 内容摘要：图像分类、目标检测、图像分割、人脸识别、图像生成
  * 代码运行：
    * 作业代码：
      * [U-Net图像分割](ch03/ex_unet.py)：
        ```
        cd ch03
        python ex_unet.py
        ```