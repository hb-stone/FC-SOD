import abc
from torch.utils.data import DataLoader
import torch
from typing import Dict
from typing import ClassVar
from typing import Sequence
INDEX_TYPE = Sequence[str]


class Evaluator(object,metaclass=abc.ABCMeta):

    indexes:ClassVar[INDEX_TYPE] = None

    def __new__(cls, *args,**kwargs) -> object:
        if cls.indexes is None:
            raise NotImplementedError("indexes property shoule be defined!")
        return super().__new__(cls)

    def __init__(self, dataloader:DataLoader, \
                 device:torch.device = torch.device('cpu')):
        super(Evaluator,self).__init__()
        self._dataloader = dataloader
        self._device = device

    @abc.abstractmethod
    def eval(self) -> Dict[str, float or int]:
        pass
