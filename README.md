# PyTorch models trained on CIFAR-10 dataset
- I modified [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) official implementation of popular CNN models, and trained those on CIFAR-10 dataset.
- I changed *number of class, filter size, stride, and padding* in the the original code so that it works with CIFAR-10.
- I also share the **weights** of these models, so you can just load the weights and use them.
- The code is highly re-producible and readable by using PyTorch-Lightning.

## Statistics of supported models
| No. |     Model    | Val. Acc. | No. Params |   Size |
|:---:|:-------------|----------:|-----------:|-------:|
| 1   | vgg11_bn     |   92.09%  |  128.813 M | 491 MB |
| 2   | vgg13_bn     |   94.29%  |  128.998 M | 492 MB |
| 3   | vgg16_bn     |   93.91%  |  134.310 M | 512 MB |
| 4   | vgg19_bn     |   93.80%  |  139.622 M | 533 MB |
| 5   | resnet18     |   93.33%  |   11.174 M |  43 MB |
| 6   | resnet34     |   92.92%  |   21.282 M |  81 MB |
| 7   | resnet50     |   93.86%  |   23.521 M |  90 MB |
| 8   | densenet121  |   94.14%  |    6.956 M |  27 MB |
| 9   | densenet161  |   94.24%  |   26.483 M | 102 MB |
| 10  | densenet169  |   94.00%  |   12.493 M |  48 MB |
| 11  | mobilenet_v2 |   94.17%  |    2.237 M |   9 MB |
| 12  | googlenet    |   92.73%  |    5.491 M |  21 MB |
| 13  | inception_v3 |   93.76%  |   21.640 M |  83 MB |

## How to use pretrained models

**Automatically download and extract the weights from Box (2.39 GB)**
```python
python cifar10_download.py
```
Or use [Google Drive](https://drive.google.com/file/d/11DDSbPqFXLzooIv6YPmXuKRIZJ24808g/view?usp=sharing) backup link (you have to download and extract manually)

**Load model and run**
```python
from cifar10_models import *

# Untrained model
my_model = vgg11_bn()

# Pretrained model
my_model = vgg11_bn(pretrained=True)
```

If you use your own images, all models expect data to be in range [0, 1] then normalize by
```python
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
```

## How to train models from scratch
Check the `cifar10_train.py` to see all available hyper-parameter choices.
To reproduce the same accuracy use the default hyper-parameters

`python cifar10_train.py --classifier resnet18 --gpu '0,'`

## How to test trained models
`python cifar10_test.py --classifier resnet18 --gpu '0,'`

Output

`TEST RESULTS
{'Accuracy': 93.33}`

## Check the TensorBoard logs
To see the training progress, cd to the `tensorboard_logs` and run TensorBoard there

`tensorboard --logdir=. --port=YOUR_PORT_NUMBER`

Then go to
`http://localhost:YOUR_PORT_NUMBER`

## Requirements
**Just to use pretrained models**
- pytorch = 1.5.0

**To train & test**
- torchvision = 0.6.0
- tensorboard = 2.2.1
- pytorch-lightning = 0.7.6