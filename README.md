# PyTorch models trained on CIFAR-10 dataset
- I modified [TorchVision](https://pytorch.org/docs/stable/torchvision/models.html) official implementaion of popular CNN models, and trained those on CIFAR-10 dataset.
- I changed *number of class, filter size, stride, and padding* in the the original code so that it works with CIFAR-10.
- I also share the **weights** of these models, so you can just load the weight and use them.

## Accuracy of supported models
| Model    | Test Accuracy |
|----------|---------------|
| vgg11_bn | 92.61%        |
| vgg13_bn | 94.27%        |
| vgg16_bn | 94.07%        |
| resnet18 | 93.48%        |

I will add more models...

## How To Use
```python
from models import *

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
- Learning rate: 0.05, reduce by factor 0.1 every 200 epochs
- Weight decay: 0.001

