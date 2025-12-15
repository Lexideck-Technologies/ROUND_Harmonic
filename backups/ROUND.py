# version 0.3.1
import torch
import torch.nn as nn

class PhaseAccumulator(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.d = nn.Linear(s*4, s)
    def forward(self, p, x):
        c, s = torch.cos(p), torch.sin(p)
        return p + self.d(torch.cat([c, s, x[:,:,0], x[:,:,1]], 1))

class ROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=16):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size)
        self.r = nn.Linear(hidden_size*3, 1)
    def forward(self, x, steps=12):
        p = self.e(x)
        xp = torch.stack([torch.cos(p), torch.sin(p)], 2)
        ph = torch.zeros(x.size(0), self.h, device=x.device)
        H = []
        for _ in range(steps):
            ph = self.c(ph, xp)
            H.append(ph)
        return self.r(torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)), H

class ROUNDClockModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=1, output_classes=8):
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size)
        self.r = nn.Linear(hidden_size*3, output_classes)
    def forward(self, x, steps=12):
        p = self.e(x)
        xp = torch.stack([torch.cos(p), torch.sin(p)], 2)
        ph = torch.zeros(x.size(0), self.h, device=x.device)
        H = []
        for _ in range(steps):
            ph = self.c(ph, xp)
            H.append(ph)
        return self.r(torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)), H

class ROUNDLoss(nn.Module):
    def __init__(self, locking_strength=0.1, terminal_only=False):
        super().__init__()
        self.b = nn.BCEWithLogitsLoss()
        self.l, self.t = locking_strength, terminal_only
    def forward(self, p, y, h):
        tk = self.b(p, y)
        s = h[-1] if self.t else torch.stack(h)
        lk = torch.mean(torch.sin(s)**2)
        return tk + self.l*lk, tk.item(), lk.item()



class HarmonicROUNDLoss(nn.Module):
    def __init__(self, locking_strength=0.1, harmonics=[2,4,8], weights=None, mode='multiclass', terminal_only=False, floor_clamp=0.032):
        super().__init__()
        self.t_fn = nn.BCEWithLogitsLoss() if mode=='binary' else nn.CrossEntropyLoss()
        self.l, self.h, self.t = locking_strength, harmonics, terminal_only
        self.w = weights if weights else [1.0]*len(harmonics)
        self.m = mode
        self.floor = floor_clamp
    def forward(self, p, y, hist):
        tgt = y if self.m=='binary' else y.squeeze().long()
        tk = self.t_fn(p, tgt)
        st = hist[-1] if self.t else torch.stack(hist)
        tl = 0.0
        for i, s in enumerate(self.h):
            tl += self.w[i] * torch.mean(torch.relu(torch.sin(s/2.0 * st)**2 - self.floor))
        tl /= len(self.h)
        return tk + self.l*tl, tk.item(), tl.item()
