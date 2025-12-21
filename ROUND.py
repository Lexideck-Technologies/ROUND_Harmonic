# version 0.8.0 - "The Frozen Basin"
import torch
import torch.nn as nn

class PhaseAccumulator(nn.Module):
    def __init__(self, s, spinor=True):
        super().__init__()
        # Spin-1 (Vector) vs Spin-1/2 (Spinor)
        self.spinor = spinor
        # Input dim: Spin1(2) + [Spinor(2) if True] + Input(2)
        # 6 if Spinor, 4 if Vector
        in_dim = 6 if spinor else 4
        self.d = nn.Linear(s * in_dim, s)

    def forward(self, p, x):
        # Spin-1 Features (Vector / 360 deg)
        c1, s1 = torch.cos(p), torch.sin(p)
        
        features = [c1, s1]
        
        if self.spinor:
            # Spin-1/2 Features (Spinor / 720 deg)
            p_half = 0.5 * p
            c_half, s_half = torch.cos(p_half), torch.sin(p_half)
            features.extend([c_half, s_half])
            
        features.append(x[:,:,0])
        features.append(x[:,:,1])
        
        # Concat: [Spin1, (Spinor), Input]
        # Concat: [Spin1, (Spinor), Input]
        return p + self.d(torch.cat(features, 1))

class WobblePhaseAccumulator(nn.Module):
    def __init__(self, s, spinor=True):
        super().__init__()
        self.spinor = spinor
        # Input dim: Spin1(2) + [Spinor(2) if True] + Wobble(2) + Input(2)
        # 8 if Spinor, 6 if Vector
        in_dim = 8 if spinor else 6
        self.d = nn.Linear(s * in_dim, s)
        self.d_w = nn.Linear(s * in_dim, s)
        self.coupling = 0.0

    def forward(self, p, x, w):
        # Spin-1 Features (Vector / 360 deg)
        c1, s1 = torch.cos(p), torch.sin(p)
        cw, sw = torch.cos(w), torch.sin(w)
        
        features = [c1, s1]
        
        if self.spinor:
            # Spin-1/2 Features (Spinor / 720 deg)
            p_half = 0.5 * p
            c_half, s_half = torch.cos(p_half), torch.sin(p_half)
            features.extend([c_half, s_half])
            
        features.extend([cw, sw])
        features.append(x[:,:,0])
        features.append(x[:,:,1])
        
        flat_features = torch.cat(features, 1)
        
        # Intrinsic Annealing: Wobble when High Potential (Barrier)
        # Potential Proxy: 1 - cos(p) (0 at Well, 2 at Barrier)
        # If coupled, add impulse to w
        w_impulse = 0.0
        if self.coupling > 0:
            potential = 1.0 - c1
            # Push w: simple additive drive
            w_impulse = self.coupling * potential
            
        # Return new_p, new_w
        return p + self.d(flat_features), w + self.d_w(flat_features) + w_impulse

class ROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=1, spinor=False, use_raw_phase=True, spin_factor=0.5):
        super().__init__()
        self.h = hidden_size
        self.use_raw_phase = use_raw_phase
        self.spin_factor = spin_factor
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size, spinor=spinor)
        self.spinor = spinor
        # Readout: [Cos, Sin] + [CosS, SinS if spinor] + [Ph if use_raw_phase]
        num_features = (3 if use_raw_phase else 2) + (2 if spinor else 0)
        readout_dim = hidden_size * num_features
        self.r = nn.Linear(readout_dim, 1)
    def forward(self, x, steps=12):
        p = self.e(x)
        xp = torch.stack([torch.cos(p), torch.sin(p)], 2)
        ph = torch.zeros(x.size(0), self.h, device=x.device)
        H = []
        for _ in range(steps):
            ph = self.c(ph, xp)
            H.append(ph)
        
        readout_features = [torch.cos(ph), torch.sin(ph)]
        if self.spinor:
             ph_s = self.spin_factor * ph
             readout_features.extend([torch.cos(ph_s), torch.sin(ph_s)])
        if self.use_raw_phase:
            readout_features.append(ph)
        
        return self.r(torch.cat(readout_features, 1)), H

class ROUNDClockModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=1, output_classes=8, spinor=True, use_raw_phase=True):
        super().__init__()
        self.h = hidden_size
        self.use_raw_phase = use_raw_phase
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size, spinor=spinor)
        self.spinor = spinor
        num_features = (3 if use_raw_phase else 2) + (2 if spinor else 0)
        readout_dim = hidden_size * num_features
        self.r = nn.Linear(readout_dim, output_classes)
    def forward(self, x, steps=12):
        p = self.e(x)
        xp = torch.stack([torch.cos(p), torch.sin(p)], 2)
        ph = torch.zeros(x.size(0), self.h, device=x.device)
        H = []
        for _ in range(steps):
            ph = self.c(ph, xp)
            H.append(ph)
            
        readout_features = [torch.cos(ph), torch.sin(ph)]
        if self.spinor:
             ph_half = 0.5 * ph
             readout_features.extend([torch.cos(ph_half), torch.sin(ph_half)])
        if self.use_raw_phase:
            readout_features.append(ph)
            
        return self.r(torch.cat(readout_features, 1)), H

class SequentialROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=1, output_classes=1, spinor=True, use_raw_phase=True, spin_factor=0.5, wobble=False, wobble_coupling=0.0):
        super().__init__()
        self.h = hidden_size
        self.use_raw_phase = use_raw_phase
        self.spin_factor = spin_factor
        self.wobble = wobble
        self.e = nn.Linear(input_dim, hidden_size)
        
        if wobble:
            self.c = WobblePhaseAccumulator(hidden_size, spinor=spinor)
            self.c.coupling = wobble_coupling
        else:
            self.c = PhaseAccumulator(hidden_size, spinor=spinor)
            
        self.spinor = spinor
        # Readout: [Cos, Sin] + [CosS, SinS if spinor] + [CosW, SinW if wobble] + [Ph if use_raw_phase]
        num_features = 2 + (2 if spinor else 0) + (2 if wobble else 0) + (1 if use_raw_phase else 0)
        readout_dim = hidden_size * 7 if wobble else hidden_size * num_features
        self.r = nn.Linear(readout_dim, output_classes)
        
    def forward(self, x):
        # x: [Batch, Seq, InputDim]
        B, S, D = x.shape
        ph = torch.zeros(B, self.h, device=x.device)
        wb = torch.zeros(B, self.h, device=x.device) if self.wobble else None
        
        prev_xt = None
        H = []
        for t in range(S):
            xt = x[:, t, :]
            pt = self.e(xt)
            xpt = torch.stack([torch.cos(pt), torch.sin(pt)], 2)
            
            if self.wobble:
                # 1. Constant Mnemonic Drift (The Clock)
                # Helps differentiate identical sequences
                wb = wb + 0.03125 # 2^-5 (Hyper-Resolution Clock)
                
                # 2. Triggered Gemination Deflection
                is_repeat = False
                if prev_xt is not None:
                    is_repeat = torch.all(torch.eq(xt, prev_xt)).item()
                
                if is_repeat:
                    # Accelerate wobble into the Z-axis to break the parity loop
                    ph, wb = self.c(ph, xpt, wb)
                else:
                    # Planar discovery, but still tracking the drift/wobble update
                    ph, wb = self.c(ph, xpt, wb)
                
                prev_xt = xt
                H.append((ph, wb))
                
                # 3. SU(2) Readout Features (7 Features)
                ph_s = self.spin_factor * ph
                readout_features = torch.cat([
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s), 
                    torch.cos(wb), torch.sin(wb),
                    ph
                ], 1)
            else:
                ph = self.c(ph, xpt)
                H.append(ph)
                ph_s = self.spin_factor * ph
                readout_features_list = [
                    torch.cos(ph), torch.sin(ph), 
                    torch.cos(ph_s), torch.sin(ph_s)
                ]
                if self.use_raw_phase:
                    readout_features_list.append(ph)
                readout_features = torch.cat(readout_features_list, 1)
            
        return self.r(readout_features), H

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
    def __init__(self, locking_strength=0.1, harmonics=[1], weights=None, mode='multiclass', terminal_only=False, wobble_gravity=0.0):
        super().__init__()
        self.t_fn = nn.BCEWithLogitsLoss() if mode=='binary' else nn.CrossEntropyLoss()
        self.l, self.h, self.t = locking_strength, harmonics, terminal_only
        self.wg = wobble_gravity
        self.w = weights if weights else [1.0]*len(harmonics)
        self.m = mode

    def forward(self, p, y, hist):
        # Fix for batch size 1 scalar issue
        if self.m == 'binary':
            tgt = y
        else:
            tgt = y.view(-1).long()
            
        tk = self.t_fn(p, tgt)
        
        # Handle Wobble History
        if isinstance(hist[0], tuple):
            # Unpack (ph, wb)
            if self.t:
                st_ph, st_wb = hist[-1]
            else:
                st_ph = torch.stack([h[0] for h in hist])
                st_wb = torch.stack([h[1] for h in hist])
        else:
            st_ph = hist[-1] if self.t else torch.stack(hist)
            st_wb = None
            
        tl = 0.0
        # Phase Locking
        for i, s in enumerate(self.h):
            tl += self.w[i] * torch.mean(torch.sin(s/2.0 * st_ph)**2)
        tl /= len(self.h)
        
        # Wobble Locking (Gravity)
        wl = 0.0
        if st_wb is not None and self.wg > 0:
            # Lock Wobble to Poles (0, Pi)
            # sin^2(w) is min at 0, pi.
            wl = torch.mean(torch.sin(st_wb)**2)
            
        return tk + self.l*tl + self.wg*wl, tk.item(), tl.item()
