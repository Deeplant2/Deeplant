import torch
from torch import nn
from torch import Tensor
from numpy.typing import ArrayLike
from denseweight import DenseWeight

class DenseWeightMSELoss(nn.modules.loss._Loss):
    '''
    #1. 클래스명: DenseWeightMSELoss\n
    #2. 목적/용도: Dense Weight MSE Loss를 구현한 클래스\n 
    #3. Input parameters:\n
    alpha = Weight를 얼마나 강하게 설정할 지에 대한 값.\n
    y = Weigth를 계산할 데이터\n
    reduction = 계산한 loss를 어떤 형식으로 반환할지 정하는 값. none, mean, sum이 있음.\n
    #4. Output: loss를 계산하고 반환.
    '''
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


