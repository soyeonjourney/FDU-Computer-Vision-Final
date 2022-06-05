# Task 3: ViT

## The Structure of this Repo

```
.
├── README.md
├── checkpoints
├── data
├── get_model_params.py
├── main.py
├── models
│   ├── __init__.py
│   ├── vit.py
│   └── vit_small.py
├── requirements.txt
├── train.sh
└── utils
    └── loss_fn.py
```

## Usage

First install prerequisites by

```shell
pip install -r requirements.txt
```

Then modify `train.sh` and run the command below

```shell
nohup bash train.py > nohup.out &
```

Note: CIFAR-100 dataset will be downloaded automatically the first time the training is performed.

## Results

| model                         | accuracy(%) | accuracy@5(%) |
| ----------------------------- | ----------- | ------------- |
| ViT                           | 65.32       | 88.16         |
| ViT for small datasets        | 69.41       | 89.88         |
| ViT (ImageNet-21K pretrained) | 91.80       | 98.67         |

## Reference

1. [ViT-pytorch](https://github.com/lucidrains/vit-pytorch): Implementation of Vision Transformer, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Pytorch.
