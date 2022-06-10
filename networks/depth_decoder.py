# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
import sys
sys.path.insert(0, "../monodepth2")
from layers import *

sys.path.insert(0, "../../SparseConvNet")
sys.path.insert(0, "../SparseConvNet")
import sparseconvnet as scn


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))


        return self.outputs


class NQGN_SparseDecoder(nn.Module):


    def __init__(self):
        self.inplanes = 512
        super(NQGN_SparseDecoder, self).__init__()

        transblock = TransBasicBlockSparse
        layers = [2, 2, 2, 2]

        self.dense_to_sparse = scn.DenseToSparse(2)
        self.add = AddSparseDense()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')


        self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
        self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
        self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
        self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)


        self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
        self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
        self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
        self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)

        self.inplanes = 64
        self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)

        self.out6_conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=True)
        self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 1, True)
        self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 1, True)
        self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
        self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
        self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)

        self.sparse_to_dense = scn.SparseToDense(2, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_transpose(self, transblock, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = scn.Sequential(
                scn.SparseToDense(2,self.inplanes * transblock.expansion),
                nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
                                  kernel_size=2, stride=stride, padding=0, bias=False),
                scn.DenseToSparse(2),
                scn.BatchNormalization(planes)
            )
        elif self.inplanes * transblock.expansion != planes:
            upsample = scn.Sequential(
                scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
                scn.BatchNormalization(planes)
            )

        layers = []

        for i in range(1, blocks):
            layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))

        layers.append(transblock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes // transblock.expansion

        return scn.Sequential(*layers)

    def _make_skip_layer(self, inplanes, planes):

        layers = scn.Sequential(
            scn.NetworkInNetwork(inplanes, planes, False),
            scn.BatchNormReLU(planes)
        )
        return layers

    def _masking(self, out, crit=0.5):
        out = 1/80 + (1/0.1 - 1/80) * out
        a = out[:,:,0::2,0::2]
        b = out[:,:,0::2,1::2]
        c = out[:,:,1::2,0::2]
        d = out[:,:,1::2,1::2]

        m_max = torch.max(torch.max(torch.max(a,b),c),d)
        m_min = torch.min(torch.min(torch.min(a,b),c),d)

        mask = self.up(m_max - m_min) > crit

        return mask.type(out.dtype)


    def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
        [in0, in1, in2, in3, in4] = x

        if labels is not None:
            [mask4, mask3, mask2, mask1, mask0] = labels

        out6 = self.sigmoid(self.out6_conv(in4))

        if labels is None:
            mask4 = self._masking(out6, crit)
            if torch.all(mask4 == torch.zeros_like(mask4)):
                mask4 = (torch.rand_like(mask4) > 0.4).type(mask4.dtype)
        in4 = in4 * mask4

        in4 = self.dense_to_sparse(in4)

        x = self.deconv1(in4)
        out5 = self.sigmoid(self.sparse_to_dense(self.out5_conv(x)))


        if labels is None:
            mask3 = self.up(mask4) * self._masking(out5, crit)
            if torch.all(mask3 == torch.zeros_like(mask3)):
                mask3 = self.up(mask4 * (torch.rand_like(mask4) > 0.4).type(mask4.dtype))
        in3 = in3 * mask3

        in3 = self.dense_to_sparse(in3)

        if use_skip:
            x = self.add([in3,self.densify3(x)])

        # upsample 2
        x = self.deconv2(x)
        out4 = self.sigmoid(self.sparse_to_dense(self.out4_conv(x)))

        if labels is None:
            mask2 = self.up(mask3) * self._masking(out4, crit)
            if torch.all(mask2 == torch.zeros_like(mask2)):
                mask2 = self.up(mask3 * (torch.rand_like(mask3) > 0.4).type(mask3.dtype))
        in2 = in2 * mask2

        in2 = self.dense_to_sparse(in2)

        if use_skip:
            x = self.add([in2,self.densify2(x)])

        # upsample 3
        x = self.deconv3(x)
        out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x)))

        if labels is None:
            mask1 = self.up(mask2) * self._masking(out3, crit)
            if torch.all(mask1 == torch.zeros_like(mask1)):
                mask1 = self.up(mask2 * (torch.rand_like(mask2) > 0.4).type(mask2.dtype))
        in1 = in1 * mask1

        in1 = self.dense_to_sparse(in1)

        if use_skip:
            x = self.add([in1,self.densify1(x)])

        # upsample 4
        x = self.deconv4(x)
        out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x)))

        if labels is None:
            mask0 = self.up(mask1) * self._masking(out2, crit)
            if torch.all(mask0 == torch.zeros_like(mask0)):
                mask0 = self.up(mask1 * (torch.rand_like(mask1) > 0.4).type(mask1.dtype))
        in0 = in0 * mask0

        in0 = self.dense_to_sparse(in0)

        if use_skip:
            x = self.add([in0, self.densify0(x)])

        # final
        x = self.final_deconv(x)
        out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))


        return [out6, out5, out4, out3, out2, out1], [mask4, mask3, mask2, mask1, mask0]

# https://github.com/kashyap7x/QGN
class TransBasicBlockSparse(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlockSparse, self).__init__()
        self.conv1 = conv3x3_sparse(inplanes, inplanes)
        self.bn1 = scn.BatchNormReLU(inplanes)
        self.relu = scn.ReLU()
        if upsample is not None and stride != 1:
            self.conv2 = scn.Sequential(
                scn.SparseToDense(2,inplanes),
                nn.ConvTranspose2d(inplanes, planes,
                                  kernel_size=2, stride=stride, padding=0,
                                  output_padding=0, bias=False),
                scn.DenseToSparse(2)
            )
        else:
            self.conv2 = conv3x3_sparse(inplanes, planes, stride)
        self.bn2 = scn.BatchNormalization(planes)
        self.add = scn.AddTable()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out = self.add([out,residual])
        out = self.relu(out)

        return out


class AddSparseDense(nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)

    def forward(self, input):
        a = input[0]
        b = input[1]
        output = scn.SparseConvNetTensor()
        output.metadata = a.metadata
        output.spatial_size = a.spatial_size
        axyz = a.get_spatial_locations()
        y = axyz[:,0]
        x = axyz[:,1]
        z = axyz[:,2]

        output.features = a.features + b[z,:,y,x]
        return output

    def input_spatial_size(self,out_size):
        return out_size

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_sparse(in_planes, out_planes, stride=1):
    "3x3 sparse convolution"
    if stride == 1:
        return scn.SubmanifoldConvolution(2, in_planes, out_planes, 3, False)
    else:
        return scn.Convolution(2, in_planes, out_planes, 3, stride, False)
