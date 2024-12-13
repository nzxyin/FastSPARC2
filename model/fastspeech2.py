import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder
from .modules import VarianceAdaptor, EMAPredictor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, model_config, preprocess_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        
        self.encoder = Encoder(model_config)
        self.ema_predictor = EMAPredictor(model_config['transformer']['encoder_hidden'], 12)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.ema_linear = nn.Linear(model_config["transformer"]["encoder_hidden"], 12)

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        bn_lens=None,
        max_bn_len=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        bn_masks = (
            get_mask_from_lengths(bn_lens, max_bn_len)
            if bn_lens is not None
            else None
        )

        encoder_output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            lr_output,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            periodicity_predictions,
            bn_lens,
            bn_masks,
        ) = self.variance_adaptor(
            encoder_output,
            src_masks,
            bn_masks,
            max_bn_len,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        ema_predictions = self.ema_predictor(lr_output, bn_masks)

        return (
            lr_output,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            periodicity_predictions,
            ema_predictions,
            src_masks,
            bn_masks,
            src_lens,
            bn_lens,
        )