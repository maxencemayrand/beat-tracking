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
    
    def __init__(self, hidden_size=256, num_layers=3):
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
        
        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def forward(self, spec):
        x = self.lstm(spec)[0]
        x = self.hid_to_beat(x)
        x = F.log_softmax(x, dim=-1)
        return x
    
    def set_lr(self, lr):
        self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr
    
    def loss(self, spec, onsets, isbeat):
        output = self(spec)
        output = output[onsets == 1]
        target = isbeat[onsets == 1]
        loss = self.loss_function(output, target)
        return loss

    def loss_from_dataset(self, dataset, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        loss = 0
        with torch.no_grad():
            loss = 0
            for specs, onsets, isbeat in dataloader:
                loss += self.loss(specs, onsets, isbeat).item()
            loss /= len(dataloader)
        return loss
    
    def learn(self, specs, onsets, isbeat):
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
    
    def fit(self, dataset, validset, batch_size=1, epochs=1):
        len_dataloader = -(-len(dataset) // batch_size)  # quick ceiling function
        train_hist = np.zeros((epochs, len_dataloader, 5))
        valid_hist = np.zeros((epochs, 5))
        for e in range(epochs):
            start = time.time()
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, (spec, onsets, isbeat) in enumerate(dataloader):
                tn, fp, fn, tp, loss = self.learn(spec, onsets, isbeat)
                train_hist[e, i] = np.array([tn, fp, fn, tp, loss])
            
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
            print(f'| Epoch {e + 1:{len(str(epochs))}} | ', end='')
            print(f'TL: {loss:5.3f} | ', end='')
            print(f'VL: {vloss:5.3f} | ', end='')
            print(f'TF: {F:.3f} | ', end='')
            print(f'VF: {vF:.3f} | ', end='')
            print(f'TA: {a:.3f} | ', end='')
            print(f'VA: {va:.3f} | ', end='')
            print(f'{t / len(dataloader):.2f} s/b | {time_per_epoch} | ETA: {eta} |')
        return train_hist, valid_hist
    
    def predict(self, specs, onsets):
        """So far only works if batch_size = 1"""
        with torch.no_grad():
            output = self(specs)
            output = output[onsets == 1]
            pred_t = torch.argmax(output, dim=1)
            onsets_frames = np.argwhere(onsets.squeeze(0) == 1).squeeze(0)
            beats_frames = onsets_frames[pred_t == 1]
            pred = torch.zeros_like(onsets)
            pred[:, beats_frames] = 1
        return pred
    
    def evaluate(self, specs, onsets, isbeat):
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
        for p in self.parameters():
            p.requires_grad = False
            
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    
class ToTensor(object):
    
    def __init__(self, device):
        self.device = device
    
    def __call__(self, audiobeats):
        spec_np = audiobeats.get_spec()
        onsets_np, isbeat_np = audiobeats.get_onsets_and_isbeat()
        
        # Normalize to [-1, 1]
        spec_np = 2 * (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min()) - 1
        spec = torch.tensor(spec_np.T, dtype=torch.float, device=self.device)
        
        onsets = torch.zeros(spec.shape[0], dtype=torch.long, device=self.device)
        isbeat = torch.zeros(spec.shape[0], dtype=torch.long, device=self.device)
        
        onsets[onsets_np] = 1
        isbeat[onsets_np[isbeat_np == 1]] = 1
        
        return spec, onsets, isbeat
    
