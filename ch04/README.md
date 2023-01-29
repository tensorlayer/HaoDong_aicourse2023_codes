# Usage

First, download the aligned face images from [google](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8) or [baidu](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F) to a `data` folder.

Please place dataset 'img_align_celeba.zip' under 'data/celebA/' by default.

Your directory structure should look like this:

```
ch04/
    └── data.py
    └── dcgan_model.py
    └── ex2_dcgan_train.py
    └── README.md
    └── data
          └── celebA
            └── img_align_celeba
                └── xxxx.jpg
                └── ......

```

Second, train the GAN:

    $ python ex2_dcgan_train.py
    
## Result on celebA


<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/result.png" width="90%" height="90%"/>
</div>
</a>
