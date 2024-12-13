import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from transformer.Layers import ConvNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.periodicity_predictor = VariancePredictor(model_config)
        # self.ema_predictor = VariancePredictor(model_config, 12)

    def forward(
        self,
        x,
        src_mask,
        bn_mask=None,
        max_len=None,
        duration_target=None,
        pitch_control=1.0,
        energy_control=1.0,
        duration_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)

        if duration_target is not None:
            x, bn_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * duration_control),
                min=0,
            )
            x, bn_len = self.length_regulator(x, duration_rounded, max_len)
            bn_mask = get_mask_from_lengths(bn_len)
        pitch_prediction = self.pitch_predictor(x, bn_mask) * pitch_control
        energy_prediction = self.energy_predictor(x, bn_mask) * energy_control
        periodicity_prediction = F.sigmoid(self.periodicity_predictor(x, bn_mask))
        # ema_prediction = self.ema_predictor(x, bn_mask)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            periodicity_prediction,
            # ema_prediction,
            bn_len,
            bn_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        bn_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            bn_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(bn_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, bn_len = self.LR(x, duration, max_len)
        return output, bn_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        # self.output_size = output_size

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.filter_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class EMAPredictor(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        dropout=0.2
    ):

        super(EMAPredictor, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv(
                    n_input_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=int((postnet_kernel_size - 1) / 2),
                ),
                nn.ReLU(),
                nn.LayerNorm(postnet_embedding_dim),
                nn.Dropout(dropout),
            )
        )

        for i in range(1, postnet_n_convolutions):
            self.convolutions.append(
                nn.Sequential(
                    Conv(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=int((postnet_kernel_size - 1) / 2),
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(postnet_embedding_dim),
                    nn.Dropout(dropout),
                    
                )
            )

        self.linear = nn.Linear(postnet_embedding_dim, n_output_channels)

    def forward(self, x, mask):
        # x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
        x = self.linear(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        # x = x.contiguous().transpose(1, 2)
        return x