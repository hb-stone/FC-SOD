"""
the package store the loss that user defined
"""

from torch import nn
import torch.nn.functional as F
import torch


def SwiftCrossEntropyLoss(pred:torch.Tensor, target:int or float or torch.Tensor,**kwargs):
    if isinstance(target, torch.Tensor):
        if isinstance(target, torch.LongTensor):
            return F.cross_entropy(pred,target,**kwargs)
        else:
            return F.cross_entropy(pred, target.long(), **kwargs)
    else:
        shape = list(pred.shape[:-1])
        target = torch.LongTensor([int(target)]).expand(shape).to(pred.device)
        return F.cross_entropy(pred, target, **kwargs)


class BCELossWithLogits_Mask(nn.Module):

    def __init__(self,ignore_value,**kwargs):
        self.ignore_value = ignore_value
        self.kwargs = kwargs
        super().__init__()

    def forward(self, pred:torch.Tensor, target:torch.Tensor or float or int, ignore_index=None):
        if isinstance(target, (float, int)):
            target = torch.FloatTensor([target]).view(*[1 for _ in range(pred.dim())]).expand_as(pred).to(pred.device)
            if ignore_index is not None:
                filter_index = 1 - ignore_index.byte()
                filter_index = filter_index.to(torch.bool)
                pred = pred[filter_index]
                target = target[filter_index]
                del filter_index
            return F.binary_cross_entropy_with_logits(pred, target, **self.kwargs)
        if target.dim() == 3 and pred.dim() == 4:
            target.unsqueeze_(dim=1)
        if ignore_index is None:
            filter_index = (target != self.ignore_value)
        else:
            filter_index = 1 - ignore_index.byte()
            filter_index = filter_index.to(torch.bool)
        pred = pred[filter_index]
        target = target[filter_index]
        del filter_index
        return F.binary_cross_entropy_with_logits(pred,target,**self.kwargs)