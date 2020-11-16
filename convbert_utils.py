import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torch.nn.modules.conv import _ConvNd, _size_1_t, _single, Tensor


class Conv1d(_ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, False,
                                     _single(0), groups, bias, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups,
                                   self.padding_mode)


def conv1d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        padding_mode="zeros"):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    if padding == "same":
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)

        if padding_rows > 0:
            if padding_mode == "zeros":
                input = F.pad(
                    input,
                    [padding_rows // 2, padding_rows - padding_rows // 2],
                    mode="constant",
                    value=0)
            else:
                input = F.pad(
                    input,
                    [padding_rows // 2, padding_rows - padding_rows // 2],
                    mode=padding_mode)
        padding = (0, )

    return F.conv1d(input,
                    weight,
                    bias,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups)


class SeparableConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: typing.Union[str, int] = 0,
                 dilation: int = 1,
                 use_bias: bool = True):
        super().__init__()
        self.use_bias: bool = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1))
        else:
            self.register_parameter('bias', None)

        self.depthwise = Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=in_channels,
                                bias=False)

        self.pointwise = Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                bias=False)

    def forward(self, inputs):
        out = self.pointwise(self.depthwise(inputs))
        if self.bias is not None:
            out += self.bias
        return out
