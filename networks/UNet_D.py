import torch.nn as nn
import torch.nn.functional as F
import torch
from config import Configuration



Conv2d = nn.Conv2d


class Classifier_Module(nn.Module):


    def __init__( self, in_channel, num_classes, dilation_series, padding_series ):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                Conv2d(in_channel, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True)
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward( self, x ):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class FCDiscriminator(nn.Module):

    def __init__( self, ndf=64 ):
        super(FCDiscriminator, self).__init__()

        self.conv1 = Conv2d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.ASPP = Classifier_Module(ndf * 8, 64, [6, 12, 18, 24], [6, 12, 18, 24])
        self.linear = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.trans_modules = nn.ModuleList()
        self.trans_modules.append(nn.Sequential(
            Conv2d(ndf * 4,64,1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))
        self.trans_modules.append(nn.Sequential(
            Conv2d(ndf * 2, 64, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))
        self.trans_modules.append(nn.Sequential(
            Conv2d(ndf, 32, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))

        self.concat_trans_modules = nn.ModuleList()
        self.concat_trans_modules.append(nn.Sequential(
            Conv2d(128,64,3,1,1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))
        self.concat_trans_modules.append(nn.Sequential(
            Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))
        self.concat_trans_modules.append(nn.Sequential(
            Conv2d(64+32, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ))
        self.score = Conv2d(64,1,1)


    def freeze_encoder_bp(self,is_freeze:bool):
        for module in (self.conv1,self.conv2,self.conv3,self.conv4,self.ASPP,self.linear):
            for p in module.parameters():
                p.requires_grad_(not is_freeze)


    def freeze_bp(self, is_freeze: bool ):
        for p in self.parameters():
            p.requires_grad_(not is_freeze)

    def forward(self, x, is_only_encoder = False):
        size = x.shape[-2:]
        out_list = []
        x = self.conv1(x)
        x = self.leaky_relu(x)
        out_list.append(x)
        x = self.conv2(x)
        conv2_out = self.leaky_relu(x)
        out_list.append(conv2_out)
        x = self.conv3(conv2_out)
        conv3_out = self.leaky_relu(x)
        out_list.append(conv3_out)

        x = self.conv4(conv3_out)
        conv4_out = self.leaky_relu(x)
        x = self.ASPP(conv4_out)
        aspp_out = self.leaky_relu(x)
        gsp_out = torch.sum(aspp_out,[-1,-2])
        gsp_out = gsp_out.squeeze()
        out = self.linear(gsp_out)
        if is_only_encoder:
            del out_list
            return out
        x1 = aspp_out

        for sub_out,concat_trans_module,trans_module\
                in zip(reversed(out_list),self.concat_trans_modules,self.trans_modules):
            x2 = trans_module(sub_out)
            x1 = F.interpolate(x1,size=x2.shape[-2:],mode='bilinear',align_corners=True)
            x = torch.cat([x1,x2],dim=1)
            x1 = concat_trans_module(x)

        full_out = self.score(x1)
        full_out = F.interpolate(full_out,size=size,mode='bilinear',align_corners=True)
        return out,full_out


if __name__ == '__main__':
    net = FCDiscriminator()
    x = torch.rand((10, 1, 321, 321))
    out,full_out = net(x)
    print(out.shape,full_out.shape)
