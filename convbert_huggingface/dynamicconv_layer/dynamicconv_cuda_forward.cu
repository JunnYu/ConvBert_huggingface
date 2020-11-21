
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dynamicconv_cuda.cuh"

std::vector<at::Tensor> dynamicconv_cuda_forward(at::Tensor input, at::Tensor weight, int padding_l) {

    at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = weight.size(1);
    const auto filterSize = weight.size(2);

    const auto numFiltersInBlock = numFeatures / numHeads;
    const dim3 blocks(minibatch, numFeatures);

    auto output = at::zeros_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();

    switch(filterSize) {

        case 3:

            if (padding_l == 1) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<3, 32, 1, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 2) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<3, 32, 2, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 5:

            if (padding_l == 2) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<5, 32, 2, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 4) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<5, 32, 4, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 7:

            if (padding_l == 3) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<7, 32, 3, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 6) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<7, 32, 6, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 9:

            if (padding_l == 4) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<9, 32, 4, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 8) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<9, 32, 8, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 15:

            if (padding_l == 7) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<15, 32, 7, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 14) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<15, 32, 14, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 31:

            if (padding_l == 15) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<31, 32, 15, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 30) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<31, 32, 30, scalar_t>
                    <<<blocks, 32, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 63:

            if (padding_l == 31) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<63, 64, 31, scalar_t>
                    <<<blocks, 64, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 62) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<63, 64, 62, scalar_t>
                    <<<blocks, 64, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 127:

            if (padding_l == 63) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<127, 128, 63, scalar_t>
                    <<<blocks, 128, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 126) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<127, 128, 126, scalar_t>
                    <<<blocks, 128, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        case 255:

            if (padding_l == 127) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<255, 256, 127, scalar_t>
                    <<<blocks, 256, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            if (padding_l == 254) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamicconv_forward", ([&] {
                    dynamicconv_forward_kernel<255, 256, 254, scalar_t>
                    <<<blocks, 256, 0, stream>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            minibatch,
                            sequenceLength,
                            numFeatures,
                            numFiltersInBlock,
                            numHeads,
                            output.data<scalar_t>());
                }));
            } else

            {
                std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
            }
            break;


        default:
            std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
    }

    return {output};
}
