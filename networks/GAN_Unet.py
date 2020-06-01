from networks.Generator import Generator
from networks.UNet_D import FCDiscriminator
import torch
import torch.nn as nn

class GAN(object):

    def __init__(self,device=torch.device('cpu'),G_pth_path=None):
        self.generator:Generator = Generator()
        self.discriminator = FCDiscriminator()
        if G_pth_path is not None:
            self._load_generator_weight(G_pth_path)
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
        self.train()

    def train(self):
        self.generator.train()
        self.discriminator.train()
        return self

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        return self

    def to(self,*args,**kwargs):
        self.discriminator.to(*args,**kwargs)
        self.generator.to(*args,**kwargs)
        return self

    def _load_generator_weight(self, pth_path):
        return self.generator.load_weight(pth_path)

    def forward_G(self,x):
        return self.generator(x)

    def forward_D(self,x,is_only_encoder = False):
        return self.discriminator(x,is_only_encoder)


    def apply(self,callback):
        self.discriminator.apply(callback)
        self.generator.apply(callback)
