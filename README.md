# A Comparative study of Architectures for Performing Semantic Segementation of 2D images on PASCAL VOC dataset
Semantic Segmentation involves understanding the im-
age on a pixel-by-pixel level i.e. to assign a class label to
every pixel in the image. We experiment with different ar-
chitectures to perform segmantic segmentation of images on
the PASCAL VOC 2012 [3] dataset.
We implement the Fully Convolutional Networks (FCN)
by Long et al.[7] as our baseline method for performing se-
mantic segmentation. We perform various experiments with
the number and position of skip connections and adding dif-
ferent layers to aggregate more context information.
We then implement an Improved Fully Convolutional
Network (IFCN) architecture as suggested in the work of
Shuai et al. [8] which introduces a context network that
progressively expands the receptive fields of feature maps.
In addition, dense skip connections are added so that the
context network can be effectively optimized and fuses rich-
scale context to make reliable predictions, which has proven
to show significant improvements in segmentation on the
PASCAL VOC 2012 [3] dataset.
We also modify the U-Net architecture for multi-class
semantic segmentation with pre-trained weights from the
VGG-16 architecture trained on the ImageNet dataset.

This repository is organized as follows:
# 1.Models:
This folder contains following Models implemented:
FCN 32s 
FCN 16 s
FCN 8s
IFCN(Improved FCN)
U-Net(pre-trained on vgg16)
Work is in progress for 
Resnet
Dilated Densenets

# 2.code 
models.py--> architecture of all models
train.py--> training file(softamx loss used)
evluate.py--> uses Mean IOU 
infernece.py--> called by evaluate.py

# 3. Project report
this is a comprehensive report detailing all the analysis and discussing my work. Have a look at it if you would like to know in more detail about the project

# 4. utils
Contains all the utility folders

# 5. train-text files
these contain all the training and the testing files

