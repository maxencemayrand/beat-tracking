import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime

from . import utils
from . import constants

class BeatFinder(nn.Module):
    r"""A recurrent neural network to predict which onsets in a piece of music
    are beats. These onsets are then used to generate a beat track with a
    standard beat tracking algorithm. The preselection made by the neural
    network helps the beat tracking algorithm generate beats that are closer to
    the ground truth.

    The recurrent neural network is a binary classifier which takes as input a
    time slice of the spectrogram of the audio and try to predict if it is a
    beat. The architecture is a sequence of LSTM cells (3 by default) followed
    by a fully connected layer with 2 outputs. The loss is only computed on the
    onsets.

    In summary, it takes as input a spectrogram of the audio (viewed as a time
    series of 1d arrays of frequencies), a list of which time slices in the
    spectrogram are onsets, and it tries to predict which of those onsets are
    beats.
    """

    def __init__(self, hidden_size=256, num_layers=3, lr=0.0005):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                        constants.nb,
                        hidden_size,
                        num_layers,
                        bidirectional=True,
                        dropout=0.2,
                        batch_first=True)
        self.hid_to_beat = nn.Linear(2 * hidden_size, 2)
        self.hidden = None

        self.loss_function = nn.NLLLoss()

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(),
                            lr=self.lr, eps=1e-6, weight_decay=1e-5,
                            amsgrad=True)

    def forward(self, spec):
        x = self.lstm(spec)[0]
        x = self.hid_to_beat(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def set_lr(self, lr):
        r"""Set the learning rate of the Adam optimizer.

        Argument:
            lr (float): learning rate.
        """
        self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr

    def learn(self, specs, onsets, isbeat):
        r"""Makes one step of the optimizer and returns the loss and the true/
        false positives/negatives. The loss is computed with binary cross
        entropy only on the inputs that are onsets.

        Arguments:
            spec (numpy array): Spectrogram of the audio sample.
            onsets (numpy array): Array of onsets in units of frames.
            isbeat (numpy array): `isbeat[i] = 1` if `onsets[i]` is a beat and
                `0` otherwise.

        Returns:
            tn (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            tp (int): Number of true positives.
            loss (float): Loss.
        """

        self.optimizer.zero_grad()
        output = self(specs)
        output = output[onsets == 1]
        target = isbeat[onsets == 1]
        loss = self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            predic = torch.argmax(output, dim=1)
            tn = torch.sum((predic == 0) & (target == 0)).item()
            fp = torch.sum((predic == 1) & (target == 0)).item()
            fn = torch.sum((predic == 0) & (target == 1)).item()
            tp = torch.sum((predic == 1) & (target == 1)).item()
            loss = loss.item()

        return tn, fp, fn, tp, loss

    def fit(self, dataset, validset, batch_size=1, epochs=1, early_stop=None):
        r"""Learn over a whole dataset for multiple epochs.

        Arguments:
            dataset (AudioBeatsDataset): Dataset to learn from.
            validset (AudioBeatsDataset): Validation set to be evaluated at
                each epoch for comparison.
            batch_size (int): Size of the minibatches.
            epochs (int): Number of epochs.
            early_stop (None or float): If float, the learning will stop when
                the average training loss over 20 minibatches is less than
                `early_stop`. This is because the loss tends to make sudden
                jumps at random times. We set this to stop before such a jump
                occurs.
        Returns:
            train_hist (numpy array): History of train loss and true/false
                positives/negatives for each minibatch of each epoch. Shape:
                (nb of epochs, nb of minibatches, 5) where the last dimension
                is [tp, fp, fn, tp, loss].
            valid_hist (numpy array): History of validation loss and true/false
                positives/negatives for each epoch. Shape: (nb of epochs, 5)
                where the last dimension is [tp, fp, fn, tp, loss].
        """

        len_dataloader = -(-len(dataset) // batch_size)  # quick ceiling function
        train_hist = np.zeros((epochs, len_dataloader, 5))
        valid_hist = np.zeros((epochs, 5))
        stop_epochs = False
        for e in range(epochs):
            start = time.time()
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, (spec, onsets, isbeat) in enumerate(dataloader):
                tn, fp, fn, tp, loss = self.learn(spec, onsets, isbeat)
                train_hist[e, i] = np.array([tn, fp, fn, tp, loss])
                if early_stop and i >= 20 and train_hist[e, i - 20: i, 4].mean() < early_stop:
                    stop_epochs = True
                    break

            vtn, vfp, vfn, vtp, vloss = self.evaluate_from_dataset(validset)
            valid_hist[e] = np.array([vtn, vfp, vfn, vtp, vloss])
            va, vp, vr, vF = utils.measures(vtn, vfp, vfn, vtp)

            ttn, tfp, tfn, ttp = tuple(np.sum(train_hist[e, :, :4], axis=0))
            loss = np.mean(train_hist[e, :, 4])
            a, p, r, F = utils.measures(ttn, tfp, tfn, ttp)

            end = time.time()
            t = end - start
            time_per_epoch = str(datetime.timedelta(seconds=int(t)))
            eta = str(datetime.timedelta(seconds=int(t * (epochs - e - 1))))
            print(f'| {e + 1:{len(str(epochs))}} | ', end='')
            print(f'L: {loss:5.3f} {vloss:5.3f} | ', end='')
            print(f'F: {F:.3f} {vF:.3f} | ', end='')
            print(f'A: {a:.3f} {va:.3f} | ', end='')
            print(f'{t / len(dataloader):.2f} s/b | {time_per_epoch} | ETA: {eta} |')
            if stop_epochs:
                print(f'Early stop at minibatch {i}')
                break
        return train_hist, valid_hist

    def predict(self, specs, onsets):
        r""" Returns the prediction for `isbeat`, i.e. which onsets are beats.
        (Only works for a minibatch of size one.)
        """
        with torch.no_grad():
            output = self(specs)
            output = output[onsets == 1]
            pred_t = torch.argmax(output, dim=1)
            onsets_frames = np.argwhere(onsets.squeeze(0).cpu() == 1).squeeze(0)
            beats_frames = onsets_frames[pred_t == 1]
            pred = torch.zeros_like(onsets)
            pred[:, beats_frames] = 1
        return pred

    def evaluate(self, specs, onsets, isbeat):
        r"""Returns true/false positives/negatives and loss of a minibatch.

        Arguments:
            spec (numpy array): Spectrogram of the audio sample.
            onsets (numpy array): Array of onsets in units of frames.
            isbeat (numpy array): `isbeat[i] = 1` if `onsets[i]` is a beat and
                `0` otherwise.

        Returns:
            tn (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            tp (int): Number of true positives.
            loss (float): Loss.
        """
        with torch.no_grad():
            output = self(specs)
            output = output[onsets == 1]
            target = isbeat[onsets == 1]
            predic = torch.argmax(output, dim=1)

            tn = torch.sum((predic == 0) & (target == 0)).item()
            fp = torch.sum((predic == 1) & (target == 0)).item()
            fn = torch.sum((predic == 0) & (target == 1)).item()
            tp = torch.sum((predic == 1) & (target == 1)).item()
            loss = self.loss_function(output, target)

        return tn, fp, fn, tp, loss

    def evaluate_from_dataset(self, dataset, batch_size=16):
        r"""Returns true/false positives/negatives and loss of a dataset.

        Arguments:
            dataset (AudioBeatsDataset): Dataset to evaluate.
            batch_size (int): Size of minibatches (doesn't change the result,
                only the time to compute it).

        Returns:
            tn (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            tp (int): Number of true positives.
            loss (float): Loss.
        """

        dataloader = DataLoader(dataset, batch_size=batch_size)
        ttn = 0
        tfp = 0
        tfn = 0
        ttp = 0
        tloss = 0

        for specs, onsets, isbeat in dataloader:
            tn, fp, fn, tp, loss = self.evaluate(specs, onsets, isbeat)
            ttn += tn
            tfp += fp
            tfn += fn
            ttp += tp
            tloss += loss
        tloss /= len(dataloader)

        return ttn, tfp, tfn, ttp, tloss

    def freeze(self):
        r"""Freeze all parameters to stop learning.
        """
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        r"""Unfreeze parameters.
        """
        for p in self.parameters():
            p.requires_grad = True

def load(filename, device):
    model = BeatFinder()
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    model.freeze()
    return model
    
            
class ToTensor(object):
    r"""Callable object to transform an `AudioBeats` object into pytorch
    tensors to feed to `BeatFinder`.

    Argument:
        device: The device on which to put the tensors.
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, audiobeats):
        r"""Transform the audiobeats in tensors.

        Argument:
            audiobeats (AudioBeats): the `AudioBeats` object to transform.

        Returns:
            spec (2d tensor): The spectrogram of the audio sample.
            onsets (1d tensor): An array of 0s and 1s which says which frames
                are onsets.
            isbeat (1d tensor): An array of 0s and 1s which says which onsets
                are beats.
        """
        spec_np = audiobeats.get_spec()
        onsets_np, isbeat_np = audiobeats.get_onsets_and_isbeat()

        # Normalize to [-1, 1]
        spec_np = 2 * (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min()) - 1
        spec = torch.tensor(spec_np.T, dtype=torch.float, device=self.device)

        onsets = torch.zeros(spec.shape[0],
                    dtype=torch.long,
                    device=self.device)
        isbeat = torch.zeros(spec.shape[0],
                    dtype=torch.long,
                    device=self.device)

        onsets[onsets_np] = 1
        isbeat[onsets_np[isbeat_np == 1]] = 1

        return spec, onsets, isbeat
