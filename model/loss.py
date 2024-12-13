import torch
import torch.nn as nn
from .sdtw_loss import SoftDTW


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.sdtw_loss = SoftDTW(gamma=0.1, normalize=True)

    def forward(self, targets, predictions):
        (
            ema_targets,
            pitch_targets,
            periodicity_targets,
            energy_targets,
            duration_targets,
        ) = targets
        assert ema_targets.shape[2] == 12
        (
            _,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            periodicity_predictions,
            ema_predictions,
            src_masks,
            bn_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        bn_masks = ~bn_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        ema_targets = ema_targets[:, : bn_masks.shape[1], :]
        bn_masks = bn_masks[:, :bn_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        ema_targets.requires_grad = False
        periodicity_targets.requires_grad = False
        
        ema_loss = self.mse_loss(ema_predictions.masked_select(bn_masks.unsqueeze(-1)), ema_targets.masked_select(bn_masks.unsqueeze(-1)))
        periodicity_loss = self.mse_loss(periodicity_predictions.masked_select(bn_masks), periodicity_targets.masked_select(bn_masks))
        pitch_loss = self.mse_loss(pitch_predictions.masked_select(bn_masks), pitch_targets.masked_select(bn_masks))
        energy_loss = self.mse_loss(energy_predictions.masked_select(bn_masks), energy_targets.masked_select(bn_masks))
        duration_loss = self.mse_loss(log_duration_predictions.masked_select(src_masks), log_duration_targets.masked_select(src_masks))
        # print(ema_loss.shape)
        # print(self.sdtw_loss(ema_predictions, ema_targets))
        # ema_loss += torch.mean(self.sdtw_loss(ema_predictions, ema_targets))
        # periodicity_loss += torch.mean(self.sdtw_loss(periodicity_predictions, periodicity_targets))
        # pitch_loss += torch.mean(self.sdtw_loss(pitch_predictions, pitch_targets))
        # energy_loss += torch.mean(self.sdtw_loss(energy_predictions, energy_targets))
        # print(ema_loss.shape)
        total_loss = (
            ema_loss + duration_loss + pitch_loss + energy_loss + periodicity_loss
        )

        return (
            total_loss,
            ema_loss,
            periodicity_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
