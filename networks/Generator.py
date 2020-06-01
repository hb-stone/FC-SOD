from base_model.ResNet import resnet101 as Resnet
from base_model.Classifier import Classifier_Module as Classifier
import torch.nn.functional as F
import torch
import torch.nn as nn


__ALL__ = ['Generator']


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = Resnet()
        self.aspp = Classifier([6,12,18,24],[6,12,18,24],1)
        # Default initialization method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
    def load_weight(self, pth_path):
        cpu_device = torch.device('cpu')
        model_dict = self.base_model.state_dict()
        old_dict = torch.load(open(pth_path, 'rb'), map_location=cpu_device)
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        self.base_model.load_state_dict(model_dict)
        del old_dict
        del model_dict
        print(f'load {pth_path} success!')
        return True
    def forward(self, x):
        base_model_out = self.base_model(x)
        out = self.aspp(base_model_out)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear',align_corners=True)


if __name__ == '__main__':
    net = Generator()
    x = torch.rand((1,3,321,321))
    out = net(x)
    print(out.shape)
