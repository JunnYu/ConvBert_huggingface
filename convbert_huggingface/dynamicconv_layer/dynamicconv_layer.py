# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dynamicconv_cuda
from torch.autograd import Function


class dynamicconvFunction(Function):
    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = dynamicconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = dynamicconv_cuda.backward(grad_output.contiguous(),
                                            ctx.padding_l, *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None