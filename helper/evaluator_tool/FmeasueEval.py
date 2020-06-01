from .Evaluator import Evaluator
from torch.utils.data import DataLoader
import torch
from typing import Dict


class FmeasueEval(Evaluator):

    """
    make sure that pred and groud truth image's shape = [N,H,W]
    and the value should belong [0,1]
    """
    indexes = ("max-F","mean-F","precision","recall")

    def __init__(self,dataloader: DataLoader, \
                 device:torch.device = torch.device('cpu')):
        super(FmeasueEval,self).__init__(dataloader,device)

    def eval(self) -> Dict[str, float]:
        """
        calculate the F-measure score
        :return: the max F-measure score and the mean F-measure score
        """
        beta2 = 0.3
        avg_f, img_num = 0.0, 0.0
        precs = 0.0
        recalls = 0.0
        with torch.no_grad():
            for preds,labels in self._dataloader:
                preds = preds.to(self._device)
                labels = labels.to(self._device)
                b = preds.size(0)
                for i in range(b):
                    pred = preds[i]
                    gt = labels[i]
                    prec, recall = self._eval_pr(pred, gt, 256)
                    recall[recall != recall] = 0.0
                    prec[prec != prec] = 0.0
                    precs += prec
                    recalls += recall
                    img_num += 1
        if img_num != 0.0:
            f_score = (1 + beta2) * precs * recalls / (beta2 * precs + recalls)
            score = f_score / img_num
            return {
                "max-F":score.max().item(),
                "mean-F":score.mean().item(),
                "precision":precs.cpu().view(-1).numpy() / img_num,
                "recall":recalls.cpu().view(-1).numpy() / img_num,
            }
        else:
            return {
                "max-F":0.0,
                "mean-F":0.0,
                "precision":None,
                "recall":None,
            }


    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num).to(self._device), torch.zeros(num).to(self._device)
        thlist = torch.linspace(0, 1.0, num).to(self._device)
        gt = (y > thlist[num // 2])
        for i in range(num):
            y_temp = (y_pred > thlist[i])
            tp = (y_temp & gt).byte().sum().float()
            prec[i] = (tp + 1e-5 ) / (y_temp.byte().sum().float() + 1e-5)
            recall[i]= (tp + 1e-5 ) / (gt.byte().sum().float() + 1e-5)
        return prec, recall