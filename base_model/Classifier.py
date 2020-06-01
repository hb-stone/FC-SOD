from torch import nn


class Classifier_Module(nn.Module):

    def __init__( self, dilation_series, padding_series, num_classes ):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True)
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward( self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out