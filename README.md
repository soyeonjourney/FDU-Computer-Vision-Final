<h1 align = "center">Final Project</h1>

This is the final assignment of DATA130051.01 Computer Vision.

In the first part of the project, we test an open sourced Deeplab v3 model trained on the Cityscapes dataset on a driving video, and the result video is shown in the link provided below.

In the second part of the project, we train three Faster R-CNN models on the VOC 2007 and 2012 datasets, and the best one uses the
pre-trained Mask R-CNN’s backbone. By respliting the train/val set, this model finally reaches an mAP of 81.11% on the test set.

In the third part of the project, we design the Vision Transformers with the same number of parameters as our refined ResNet-18 in midterm assignment and train them on the CIFAR-100 dataset, the best model achieves accuracies of 69.41% and 89.88% for top-1 and top-5 metrics respectively, we also fine-tune the pre-trained ViT on CIFAR-100 and the results even outperform some of the original paper’s.

The usages of the codes like training and testing processes are shown in the corresponding folders.

