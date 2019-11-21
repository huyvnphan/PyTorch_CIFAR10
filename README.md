# PyTorch models trained on CIFAR-10 dataset
- I modified [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) official implementation of popular CNN models, and trained those on CIFAR-10 dataset.
- I changed *number of class, filter size, stride, and padding* in the the original code so that it works with CIFAR-10.
- I also share the **weights** of these models, so you can just load the weights and use them.

## Statistics of supported models
| No. |     Model    | Val. Acc. | No. Params |   Size |
|:---:|:------------:|:---------:|-----------:|-------:|
| 1   | vgg11_bn     |   92.61%  |  128.813 M | 491 MB |
| 2   | vgg13_bn     |   94.27%  |  128.998 M | 492 MB |
| 3   | vgg16_bn     |   94.07%  |  134.310 M | 512 MB |
| 4   | vgg19_bn     |   94.25%  |  139.622 M | 533 MB |
| 5   | resnet18     |   93.48%  |   11.174 M |  43 MB |
| 6   | resnet34     |   93.82%  |   21.282 M |  81 MB |
| 7   | resnet50     |   94.38%  |   23.521 M |  90 MB |
| 8   | densenet121  |   94.76%  |    6.956 M |  27 MB |
| 9   | densenet161  |   94.96%  |   26.483 M | 102 MB |
| 10  | densenet169  |   94.74%  |   12.493 M |  48 MB |
| 11  | mobilenet_v2 |   93.85%  |    2.237 M |   9 MB |
| 12  | googlenet    |   95.08%  |    5.491 M |  21 MB |
| 13  | inception_v3 |   95.41%  |   21.640 M |  83 MB |

## How To Use

**Download the weights**

Download weights from [Google Drive Link](https://drive.google.com/drive/folders/15jBlLkOFg0eK-pwsmXoSesNDyDb_HOeV?usp=sharing), and put the weights in **models/state_dicts/** folder.

```python
from cifar10_models import *

# Untrained model
my_model = vgg11_bn()

# Pretrained model
my_model = vgg11_bn(pretrained=True)
```

**Remember to normalize data before feeding to model**
```python
transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
```

## Training Hyper-paramters
- Batch size: 256
- Number of epochs: 600
- Learning rate: 0.05, multiply by factor 0.1 every 200 epochs
- Weight decay: 0.001
- Nesterov SGD optimizer with momentum = 0.9
