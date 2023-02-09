# 🍀 基于Torchvision中的经典模型调用 :yum:


**This code calls the models in Torchvision, and the classification network topic framework is derived from Torchvision.**
基于torchvision中的模型写了的调用接口，对于刚开始学习深度学习的人可以以此来跑各个经典模型的demo，如需了解具体网络框架和代码细节可以深入torchvision源码。例如以下经典的分类模型和分割模型都可以调用：
```python
from .alexnet import *
from .resnet import *
from .vgg import *
from .squeezenet import *
from .inception import *
from .densenet import *
from .googlenet import *
from .mobilenet import *
from .mnasnet import *
from .shufflenetv2 import *
from . import segmentation
from . import detection
from . import video
```
![image info](image/classifier.png) 
```python
#Several classification frameworks are available
AlexNet、densenet121、densenet169、densenet201、densenet161、GoogLeNet、Inception3、mnasnet0_5、mnasnet0_75、mnasnet1_0、mnasnet1_3、MobileNetV2、resnet18、resnet34、resnet50、resnet101、resnet152、resnext50_32x4d、resnext101_32x8d、wide_resnet50_2、wide_resnet101_2、vgg11、vgg13、vgg16、vgg19、vgg11_bn、vgg13_bn、vgg16_bn、vgg19_bn...........
```
*The above is the classic network framework available within the models, and only for the classification networks within.This code is can take transfer learning , download the ImageNet pre trained initial model and then transfer learning  in your code, and can be frozen convolution training only full connection layer, or global training, we only use the convolution of the classic network layer, and then the convolution results set on our lightweight classifier，*


## :rainbow: Train on our datasets
We used this classifier to predict the gender of the chicken, and we used vgg16,vgg16_bn,vgg19,vgg19_bn,resnet18,resnet34、densenet101 made a comparison。You can get our dataset [here](https://drive.google.com/open?id=1eGq8dWGL0I3rW2B9eJ_casH0_D3x7R73 "dataset")
我们用的是自己制作的一个鸡性别分类数据集，可以在[谷歌云盘](https://drive.google.com/open?id=1eGq8dWGL0I3rW2B9eJ_casH0_D3x7R73 "dataset")中获得我们的数据，

**Some sample images from Our dataset:**
![image info](image/dataset.jpg) 

## :star2: Train on Custom Dataset
如果需要在你自己的数据集上运行，只需要如下面的结构存放即可，n代表你是几分类任务，例如train下label_1下是存放第一个类别的所有训练图像，以此类推
```
-your datasets
 |--train
 |   |--label_1
 |   |--label_2
 |   |--label_n
 |--test or Val
     |--label_1
     |--label_2
     |--label_n
```
Your data set needs to look like the file structure above.And if you're not dichotomous, change the last output dimension from 2 to n。
 **Then execute the following command**
 
 `python train.py --data_directory=your dataset --arch=vgg16`
 data_directory用以指明你的数据集路径地址，arch为选择的模型名
if you want to train on resnet or densenet and other, you can change the --arch=vgg16 to --arch=resnet34 or -- arch=densenet101 or other
## Visualization of Training Process
Use tensorboard for visualization. After training, you can enter the following command for visualization.

**Then visit the page that pops up on the command line,the following image will appear**

`tensorboard --logdirs=logs`

![image info](image/vis.jpg)

**Visit the above page and download the corresponding CSV, then plot the training process according to csv_plot.py：**
![image info](image/Plot.jpg)
**You can adjust the parameters to make the training process more beautiful**
