import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure
from torch.nn.functional import mse_loss
from metrics import phase_metric_torch, perp_loss
from models.model_utils import to_complex, coil_combine

"""
A collection of utilities that can be used during training such as loss functions,
early stoppers, etc.
"""

class EarlyStopper:
    """
    Ends training if validation loss increases for [patience] epochs (within 
    [min_delta] margin of error).
    """
    def __init__(self, patience=8, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def loss_criterion(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str, return_ssim=True, 
    sc_target=True, smap=None, weight_balance=None
):
    """
    General loss function that returns computed loss given prediction and target
    inputs. Must specify loss_type. Expects pred and target to be real/imaginary
    channels separated (each as torch.float32 type).
    """
    if loss_type.lower() == "mse":
        loss = mse_loss(pred, target)

        if return_ssim:
            if sc_target:
                pred = torch.abs(to_complex(pred))
                target = torch.abs(to_complex(target))
            else:
                pred = torch.abs(coil_combine(to_complex(pred), smap))
                target = torch.abs(coil_combine(to_complex(target), smap))
            data_range = torch.max(target) - torch.min(target)
            ssim = structural_similarity_index_measure(
                pred, target, data_range=data_range, reduction="elementwise_mean"
            )

    elif loss_type.lower() == "ssim":
        pred = torch.abs(to_complex(pred))
        target = torch.abs(to_complex(target))
        data_range = torch.max(target) - torch.min(target)
        ssim = structural_similarity_index_measure(
            pred, target, data_range=data_range, reduction="elementwise_mean"
        )
        loss = 1 - ssim

    
    elif loss_type.lower() == 'l1':
        pred = to_complex(pred)
        target = to_complex(target)

        loss = torch.mean(torch.abs(pred - target))

        if return_ssim:
            pred = torch.abs(coil_combine(pred, smap))
            target = torch.abs(coil_combine(target, smap))
            data_range = torch.max(target) - torch.min(target)
            ssim = structural_similarity_index_measure(
                pred, target, data_range=data_range, reduction="elementwise_mean"
            )
    
    else:
        raise Exception("That loss function is not implemented :(")

    if return_ssim:
        return loss, ssim
    else:
        return loss


class FracEarlyStopper:
    """
    Ends training if validation loss increases or only decreases by a small amount
    for [patience] epochs (within [min_delta] margin of error). Assumes our loss 
    is something that should be decreasing.
    """

    def __init__(self, patience=8, min_delta=1e-10, prev_best_loss=None, threshold=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.prev_best_loss = prev_best_loss
        self.threshold = threshold

    def early_stop(self, validation_loss):
        if self.prev_best_loss is None:
            self.prev_best_loss = validation_loss
            return False
        else:
            # If validation loss icnreases, this will be negative.
            # If validation loss decreases by a very small amount, this will be
            # a very small fraction.
            improvement_frac = 1 - (validation_loss) / (self.prev_best_loss + self.min_delta)

            if validation_loss < self.prev_best_loss:
                self.prev_best_loss = validation_loss

            # If you have a really small change, or your loss is increasing 
            if improvement_frac < self.threshold:
                self.counter += 1
                if self.counter == self.patience:
                    return True
                else:
                    return False
            else:
                self.counter = 0
                return False


def load_checkpoint(checkpoint_fpath, model, optimizer):
    """
    Load a saved model, optimizer, last epoch, and best validation loss from
    a model checkpoint.
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"], checkpoint["best_loss"]


def wandb_scale_img(data: np.ndarray) -> np.ndarray:
    """Scales float data to a range of 0, 1"""
    d_max = np.max(data)
    d_min = np.min(data)
    data = (data - d_min) / (d_max - d_min)
    return data
