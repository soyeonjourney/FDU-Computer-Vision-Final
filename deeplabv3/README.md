# Task 1: Semantic Segmentation on a Driving Video

<p align="center">
  <img src="https://user-images.githubusercontent.com/58239326/171879715-fd7fd789-743e-4d20-9132-6ea9c59c72cb.png" alt="Sublime's custom image", width="350"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/58239326/171880066-189f1b1d-21a1-442a-8de8-59f8933f5568.png" alt="Sublime's custom image", width="350"/><img src="https://user-images.githubusercontent.com/58239326/171880108-f993410e-cb47-43fc-87e6-81d52d91708e.png" alt="Sublime's custom image", width="350"/>
</p>

(The first image is the original image, the second image is the predicted image, and the last one is the overlap of these two images.)

The original video and the predicted videos can be found in my google drive link [here](https://drive.google.com/drive/folders/1iFgFBGFRjPXXF57c9PZRKXVIl0Xgvkqb?usp=sharing).

In this task, we use an open-sourced semantic segmentation model - "DeepLab v3" trained on the Cityscapes Dataset. And test this model using a driving video. And due to the size limit of the GitHub repo, we leave out the model. You can download this model from our releases. The structure of this repo is the full version of the repo (not the one you see here). And you can add these files to where they should exist at. 

The changes are:

Add [mrcnn_model.pth](https://github.com/Tequila-Sunrise/FDU-Computer-Vision-Final/releases/download/publish/mrcnn_model.pth) to folder pretrained_models.

## The Structure of this Repo

```
.
├── README.md
├── data
│   └── thn: The folder to place the images.
├── datasets.py
├── evaluation
│   ├── eval_on_val.py
│   └── eval_on_val_for_metrics.py
├── model
│   ├── aspp.py
│   ├── deeplabv3.py
│   └── resnet.py
├── pretrained_models
│   ├── mrcnn_model.pth
│   └── resnet: The folder to put the pretrained ResNet model.
├── train.py
├── utils
│   ├── preprocess_data.py
│   ├── random_code.py
│   └── utils.py
├── video_to_image.py
└── visualization.py
```

## Testing Process

1. Convert the video to images, and put it on folder `./data/thn` by running `video_to_image.py`. (Remember to change the repository in this file).
2. Run `visualization.py` to get the processed images, and it will be put in the folder `./training_logs/model_eval_seq_thn`.
3. Convert these images to a video by running command shown below, the 15 in this command refers to fps, you can change it according to the fps of the original video. 

```
# To get the prediction.
ffmpeg -r 15 -pattern_type glob -i './training_logs/model_eval_seq_thn/*_pred.png' -c:v libx264 out_pred15.mp4
# To get the overlayed result. (Overlay the original video with the predicted video.)
ffmpeg -r 15 -pattern_type glob -i './training_logs/model_eval_seq_thn/*_overlayed.png' -c:v libx264 out_overlay15.mp4
```

4. Then the video is generated.

## Reference

This repo is mainly derived from https://github.com/fregu856/deeplabv3.
