# PyTorch models trained on CIFAR-10 dataset
- I modified [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) official implementation of popular CNN models, and trained those on CIFAR-10 dataset.
- I changed *number of class, filter size, stride, and padding* in the the original code so that it works with CIFAR-10.
- I also share the **weights** of these models, so you can just load the weights and use them.
- The code is highly re-producible and readable by using PyTorch-Lightning.

## Statistics of supported models
| No. |     Model    | Val. Acc. | No. Params |   Size |
|:---:|:-------------|----------:|-----------:|-------:|
| 1   | vgg11_bn     |   92.39%  |   28.150 M | 108 MB |
| 2   | vgg13_bn     |   94.22%  |   28.334 M | 109 MB |
| 3   | vgg16_bn     |   94.00%  |   33.647 M | 129 MB |
| 4   | vgg19_bn     |   93.95%  |   38.959 M | 149 MB |
| 5   | resnet18     |   93.07%  |   11.174 M |  43 MB |
| 6   | resnet34     |   93.34%  |   21.282 M |  82 MB |
| 7   | resnet50     |   93.65%  |   23.521 M |  91 MB |
| 8   | densenet121  |   94.06%  |    6.956 M |  28 MB |
| 9   | densenet161  |   94.07%  |   26.483 M | 103 MB |
| 10  | densenet169  |   94.05%  |   12.493 M |  49 MB |
| 11  | mobilenet_v2 |   93.91%  |    2.237 M |   9 MB |
| 12  | googlenet    |   92.85%  |    5.491 M |  22 MB |
| 13  | inception_v3 |   93.74%  |   21.640 M |  83 MB |

## Details report
Weight and Biases' details report for this project [WandB Report](https://wandb.ai/huyvnphan/cifar10/reports/CIFAR10-Classification-using-PyTorch---VmlldzozOTg0ODQ?accessToken=9m2q1ajhppuziprsq9tlryynvmqbkrbvjdoktrz7o6gtqilmtqbv2r9jjrtb2tqq)

## How To Cite
[![DOI](https://zenodo.org/badge/195914773.svg)](https://zenodo.org/badge/latestdoi/195914773)

## How to use pretrained models

**Automatically download and extract the weights from Box (933 MB)**
```python
python train.py --download_weights 1
```
Or use [Google Drive](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view?usp=sharing) backup link (you have to download and extract manually)

**Load model and run**
```python
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Untrained model
my_model = vgg11_bn()

# Pretrained model
my_model = vgg11_bn(pretrained=True)
my_model.eval() # for evaluation
```

If you use your own images, all models expect data to be in range [0, 1] then normalized by
```python
mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]
```

## How to train models from scratch
Check the `train.py` to see all available hyper-parameter choices.
To reproduce the same accuracy use the default hyper-parameters

`python train.py --classifier resnet18`

## How to test pretrained models
`python train.py --test_phase 1 --pretrained 1 --classifier resnet18`

Output

`{'acc/test': tensor(93.0689, device='cuda:0')}`


## Requirements
**Just to use pretrained models**
- pytorch = 1.7.0

**To train & test**
- pytorch = 1.7.0
- torchvision = 0.7.0
- tensorboard = 2.2.1
- pytorch-lightning = 1.1.0
