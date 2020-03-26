## Semantic-Segmentation语义分割模型在Keras当中的实现
我将会将许多的语义分割的模型上传到这个仓库中提供给大家学习使用，包括segnet,pspnet,unet,deeplab_v3_plus等。  
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [使用方法与数据集下载 Download](#使用方法与数据集下载)
3. [训练效果 Performance](#训练效果)
4. [参考资料 Reference](#Reference)

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 使用方法与数据集下载
你可以下载后进入你所想要训练的模型的文件夹，然后运行train.py进行训练。  
在训练之前，需要先下载数据集，并将其存储到dataset中。  
大家关心的多分类的代码在Muiti_Class_deeplab_Mobile里。  

斑马线数据集：  
链接：https://pan.baidu.com/s/1uzwqLaCXcWe06xEXk1ROWw 提取码：pp6w   

VOC数据集：  
链接: https://pan.baidu.com/s/1Urh9W7XPNMF8yR67SDjAAQ 提取码: cvy2

### 训练效果
原图Before
![原图Before](/SegNet_Mobile/img/timg.jpg)  
#### SegNet_Mobile
处理后After processing
![处理后After processing](/SegNet_Mobile/img_out/timg.jpg)  
#### Unet_Mobile
处理后After processing
![处理后After processing](/Unet_Mobile/img_out/timg.jpg)  
#### pspnet_Mobile
处理后After processing
![处理后After processing](/pspnet_Mobile/img_out/timg.jpg)  
#### pspnet_Multi_Mobile
处理后After processing
![处理后After processing](/pspnet_Multi_Mobile/img_out/timg.jpg)  

### Reference
[image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras)  
