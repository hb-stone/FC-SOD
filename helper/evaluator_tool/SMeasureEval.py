from .Evaluator import Evaluator
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Dict


class SMeasureEval(Evaluator):

    indexes = ("S",)

    """
    make sure that pred and groud truth image's shape = [N,H,W]
    and the value should belong [0,1]
    """
    def __init__(self,dataloader: DataLoader, \
                 device:torch.device = torch.device('cpu')):
        super(SMeasureEval,self).__init__(dataloader,device)

    def eval(self) ->Dict[str,float]:
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            for preds,labels in self._dataloader:
                preds = preds.to(self._device)
                labels = labels.to(self._device)
                b = preds.size(0)
                for i in range(b):
                    pred = preds[i]
                    gt = labels[i]
                    y = gt.mean()
                    if y == 0:
                        x = pred.mean()
                        Q = 1.0 - x
                    elif y == 1:
                        x = pred.mean()
                        Q = x
                    else:
                        gt[gt >= 0.5] = 1
                        gt[gt < 0.5] = 0
                        Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                        if Q.item() < 0:
                            Q = torch.FloatTensor([0.0])
                    img_num += 1.0
                    avg_q += Q.item()
            avg_q /= img_num
        if img_num == 0:
            return {"S" : 0.0}
        return {"S":avg_q}

    def _S_object(self, pred, gt ):
        fg = torch.where(gt == 0.0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1.0, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object( self, pred, gt ):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region( self, pred, gt ):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1).to(self._device) * round(cols / 2)
            Y = torch.eye(1).to(self._device) * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0,cols)).to(self._device).float()
            j = torch.from_numpy(np.arange(0,rows)).to(self._device).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()

    def _dividePrediction(self, pred, X, Y ):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _divideGT( self, gt, X, Y ):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _ssim(self, pred, gt ):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q