## Semantic-Segmentation语义分割模型在Keras当中的实现
---

## 大通知！
PSPnet-Keras重制版如下：    
源码路径：https://github.com/bubbliiiing/pspnet-keras    
视频地址：https://www.bilibili.com/video/BV1bz4y1f77C 

PSPnet-Pytorch重制版如下：    
源码路径：https://github.com/bubbliiiing/pspnet-pytorch     
视频地址：https://www.bilibili.com/video/BV1zt4y1q7HH     

PSPnet-Tensorflow2重制版如下：    
源码路径：https://github.com/bubbliiiing/pspnet-tf2     
视频地址：https://www.bilibili.com/video/BV1Wh411f7NU     

Unet-Keras重制版如下：  
源码路径：https://github.com/bubbliiiing/unet-keras    
视频地址：https://www.bilibili.com/video/BV1St4y1r7hE      

Unet-Pytorch重制版如下：  
源码路径：https://github.com/bubbliiiing/unet-pytorch    
视频地址：https://www.bilibili.com/video/BV1rz4y117rR  

Unet-Tensorflow2重制版如下：  
源码路径：https://github.com/bubbliiiing/unet-tf2    

Deeplab-Keras重制版如下：  
源码路径：https://github.com/bubbliiiing/deeplabv3-plus-keras   



## 目录
1. [所需环境 Environment](#所需环境)
2. [注意事项 Attention](#注意事项)
3. [数据集下载 Download](#数据集下载)
4. [训练步骤 How2train](#训练步骤)
5. [预测步骤 How2predict](#预测步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 注意事项
该代码是我早期整理的语义分割代码，尽管可以使用，但是存在许多缺点。大家尽量可以使用重制版的代码，因为重制版的代码里面增加了很多新内容，比如添加了Dice-loss，增加了更多参数的选择，提供了VOC预训练权重等。  

在2021年1月28重新上传了该库，给代码添加了非常详细的注释，该库仍然可以作为一个语义分割的入门库进行使用。   

在使用前一定要注意根目录与相对目录的选取，这样才能正常进行训练。

## 数据集下载
斑马线数据集：  
链接：https://pan.baidu.com/s/1uzwqLaCXcWe06xEXk1ROWw 提取码：pp6w   

VOC数据集：  
链接: https://pan.baidu.com/s/1Urh9W7XPNMF8yR67SDjAAQ 提取码: cvy2

## 训练步骤
1. 准备好训练数据集，如果想要进行简单尝试，可以通过如上的斑马线数据集进行尝试；如果想要进行自己的数据集训练，可以参考制作自己的数据集的视频，进行制作
2. 在完成数据集的准备后，利用pycharm或者vscode打开对应模型的文件夹，将数据集及其对应的train.txt文件复制到datasets2文件夹中
3. 然后运行train.py进行训练。   
4. 大家关心的多分类的代码在Muiti_Class_deeplab_Mobile里。

## 预测步骤
1. 除去Muiti_Class_deeplab_Mobile可以直接运行predict.py进行预测外，其它的模型均需要先完成训练才可以预测。
2. 在完成训练后，将predict.py里面模型载入的权重更换成logs文件夹内的权值。
3. 将想要预测的图片放入img文件夹。
4. 运行predict.py即可开始预测。

## Reference
[image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)  
