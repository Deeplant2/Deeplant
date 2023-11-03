import torch
import numpy as np
import mlflow

from torch import nn
from torch import Tensor
from numpy.typing import ArrayLike
from denseweight import DenseWeight


# +
class DenseWeightMSELoss(nn.modules.loss._Loss):
    def __init__(self, alpha: float, y: ArrayLike, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.y = y
        self.dw = []
        for i in range(len(self.y[0])):
            self.dw.append(DenseWeight(alpha=alpha))
            self.dw[i].fit(y[:,i])
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight = []
        for i in range(len(self.y[0])):
            weight.append(self.dw[i](target[:,i].cpu()))
        weight = torch.tensor(weight).transpose(1,0)

        if self.reduction == 'none':
            return weight.cuda() * nn.functional.mse_loss(input, target, reduction='none')
        elif self.reduction == 'mean':
            return torch.mean(weight.cuda() * nn.functional.mse_loss(input, target, reduction='none'))
        elif self.reduction == 'sum':
            return torch.sum(weight.cuda() * nn.functional.mse_loss(input, target, reduction='none'))
        
        
# -


