from torch.nn import Module
import torch
from helper.utils import mkdirs
from helper.utils import pathjoin
import time


class ModelSaver(object):

    def __init__(self,save_interval,save_dir_path=None,save_base_name:str ='model'):
        self.__timer = 0
        self.__save_interval = save_interval
        self.__save_dir_path = save_dir_path
        if self.__save_dir_path is None:
            self.__save_dir_path = pathjoin(
                '../',
                time.strftime("%F %H-%M-%S",time.localtime())
            )
        mkdirs(self.__save_dir_path)
        self.__base_model_name = save_base_name
        self.__interval_timer = 0


    def __call__(self, model:Module, isFinal:bool = False,save_base_name = None,is_add=True):
        if is_add:
            self.__timer += 1
        if self.__timer % self.__save_interval == 0 or isFinal:
            self.__interval_timer += 1
            model_ext_name = 'final' if isFinal else str(self.__interval_timer)
            if save_base_name is None:
                save_base_name = self.__base_model_name
            model_save_path = pathjoin(
                self.__save_dir_path,
                save_base_name+'-'+model_ext_name+'.pth'
            )
            torch.save(model.state_dict(),model_save_path)

