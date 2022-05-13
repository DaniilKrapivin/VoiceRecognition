import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

def create_model(classes: int,n_epochs: int,device=False):
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0],
                            stride=model.conv1.stride[0], padding=model.conv1.padding[0])
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        *[nn.Dropout(p=0.25), nn.Linear(num_ftrs, classes)])
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3,
                                          gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    return model,optimizer,scheduler,criterion
