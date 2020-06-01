import torch
import abc
from typing import Dict

ImgsType = Dict[str, torch.Tensor]

class VisualHelper(object,metaclass=abc.ABCMeta):

    def __init__(self, catch_time_interval:int):
        super(VisualHelper,self).__init__()
        self._catch_time_interval = catch_time_interval
        self._timer = 0
        self._catch_timer = 0

    def add_timer(self):
        self._timer += 1

    def is_catch_snapshot(self):
        if self._timer % self._catch_time_interval == 0:
            self._catch_timer += 1
            return True
        return False

    @abc.abstractmethod
    def call(self, epoch: int, avg_loss:float, show_image:torch.Tensor):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    def __call__(self, epoch: int,avg_loss:float,preds:torch.Tensor):
        self.call(epoch,avg_loss,preds)






