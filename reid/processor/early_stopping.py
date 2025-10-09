import torch
import numpy as np


# EarlyStopping类，用于监控验证集的表现
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True, mode='min', restore_best_weights=False):
        """
        Initialize EarlyStopping.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            mode (str): One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored has stopped decreasing;
                        in "max" mode it will stop when the quantity monitored has stopped increasing.
            restore_best_weights (bool): Whether to restore model weights from the best epoch after early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == 'min' else -np.Inf  # Initialize to inf for 'min' mode, or -inf for 'max' mode

    def __call__(self, val_loss, model, epoch):
        """
        This function is called every epoch to check if early stopping should be triggered.
        """
        if self.mode == 'min':
            score = -val_loss  # In "min" mode, lower validation loss is better
        else:
            score = val_loss  # In "max" mode, higher validation accuracy is better

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping at epoch {epoch}')
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        # Optionally restore the best model weights
        if self.early_stop and self.restore_best_weights:
            model.load_state_dict(torch.load(f'checkpoint_epoch_{self.best_epoch}.pt'))

