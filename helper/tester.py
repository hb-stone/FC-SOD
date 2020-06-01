from networks.Generator import Generator as Net
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from helper.utils import mkdirs
from helper.utils import pathjoin
from typing import Callable
import os
LABEL_TRANS_FUNC = Callable[[torch.Tensor],torch.Tensor]
import cv2
__all__ = ['Tester']


class Tester(object):
    """
    the helper to train network model
    """
    def __init__(self,
                 dataloader: DataLoader,
                 img_save_path: str,
                 model_pth_path: str,
                 device:torch.device = torch.device('cpu'),
                 label_trans_func:LABEL_TRANS_FUNC = lambda x:x,
                 *args,
                 **kwargs) -> None:
        super(Tester,self).__init__(*args,**kwargs)
        self._dataloader = dataloader
        self._img_save_path = img_save_path
        mkdirs(img_save_path)
        self._device = device
        self._label_transform_func = label_trans_func
        model_dict = torch.load(model_pth_path,map_location='cpu')
        self._model = Net()
        self._model.load_state_dict(model_dict)
        del model_dict
        self._model = self._model.to(device)
        self._model.eval()


    def test(self) -> None:
        dataset_len = len(self._dataloader.dataset)
        pbar = tqdm(total=dataset_len)
        with torch.no_grad():
            for item in self._dataloader:
                imgs = item['image'].to(self._device)
                preds = self._model(imgs).cpu()
                preds = self._label_transform_func(preds)
                for i in range(preds.shape[0]):
                    filename = os.path.splitext(item['filename'][i])[0]+'.png'
                    filepath = pathjoin(self._img_save_path,filename)
                    cv2.imwrite(filepath, preds[i])
                    pbar.update(1)
                    pbar.set_description("Processing %s" % filename)
