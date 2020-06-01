from .Evaluator import Evaluator
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Dict


class MAEEval(Evaluator):
    """
    make sure that pred and groud truth image's shape = [N,H,W]
    and the value should belong [0,1]
    """
    indexes = ("MAE",)

    def __init__(self,dataloader: DataLoader, \
                 device:torch.device = torch.device('cpu')):
        super(MAEEval,self).__init__(dataloader,device)


    def eval(self)-> Dict[str,float]:
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            for preds,labels in self._dataloader:
                preds = preds.to(self._device)
                labels = labels.to(self._device)
                b = preds.size(0)
                for i in range(b):
                    pred = preds[i]
                    gt = labels[i]
                    mea = torch.abs(pred - gt).mean()
                    """
                    help:
                    ```
                    c = torch.from_numpy(np.array([np.inf]))
                    a = c-c
                    print(a == a)
                    ```
                    Out: tensor([0], dtype=torch.uint8)
                    """
                    if mea == mea:  # for Nan
                        avg_mae += mea
                        img_num += 1.0
        if img_num == 0.0:
            return {"MAE": float("inf")}
        avg_mae /= img_num
        return {"MAE": avg_mae.item()}


if __name__ == '__main__':

    measure_class = MAEEval(111) # ok
    # from . import FmeasueEval
    # print(type(FmeasueEval))