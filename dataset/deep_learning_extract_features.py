import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tramsforms
from torch.autograd import Variable


model = models.alexnet()
model.eval()
