from torch.nn import Module
from typing import Callable

__all__ = ['Optimizer']
LR_SCHUDUER_FUNC = Callable[[object,float,int,int],None]

class Optimizer(object):

    def __init__(self, optimizer_create_func:Callable,
                 total_iter_time: int,
                 lr_schuduer:LR_SCHUDUER_FUNC = None, step_time_interval = 1):
        self.__step_time_interval = step_time_interval
        self.__timer = 0
        self.__step_timer = 0
        self.__optimizer_create_func = optimizer_create_func
        self.__lr_schuduer = lr_schuduer
        self.__total_iter_time = total_iter_time // self.__step_time_interval
        self.__lr = None

    def __call__(self, model:Module, lr: float):
        self.__optimizer = self.__optimizer_create_func(model, lr)
        self.__optimizer.zero_grad()
        self.__lr = lr

    @property
    def step_time_interval( self ):
        return self.__step_time_interval

    def step(self):
        self.__timer += 1
        if self.__timer % self.__step_time_interval == 0:
            self.__step_timer += 1
            self.__optimizer.step()
            self.__optimizer.zero_grad()
            if self.__lr_schuduer is not None:
                self.__lr_schuduer(self.__optimizer, self.__lr, \
                                   self.__step_timer, self.__total_iter_time)
