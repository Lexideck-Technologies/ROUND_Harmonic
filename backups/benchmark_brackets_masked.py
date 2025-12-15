
# version 0.3.1 - Masked/Sequential Test
import torch,torch.nn as nn,torch.optim as optim,numpy as np,matplotlib.pyplot as plt,os,random,uuid
from ROUND import PhaseAccumulator, HarmonicROUNDLoss

if not os.path.exists('data'):os.makedirs('data')
UID=os.environ.get('ROUND_BATCH_UID',str(uuid.uuid4())[:8])
L_FILE=open(f'data/log_brackets_masked_{UID}.txt','w')
def P(s):print(s);L_FILE.write(str(s)+'\n');L_FILE.flush()
P(f"Batch UID: {UID}")
C={'task':'brackets_masked','input_dim':20,'hidden_size':32,'steps':20,'epochs':1000,'batch_size':64,'dataset_size':4000,'runs':5,'lr':0.001953125,'device':'cuda' if torch.cuda.is_available() else 'cpu'}

# --- Data Generation (Same as benchmark_brackets.py) ---
def is_balanced(seq):
    b=0
    for c in seq:
        b+=1 if c==0 else -1
        if b<0:return False
    return b==0

def generate_dyck_data(n,l):
    X,Y,n_pos,n_neg=[],[],n//2,n-(n//2)
    cp,cn=0,0
    while cp<n_pos or cn<n_neg:
        bs=100;x=torch.randint(0,2,(bs,l)).float()
        for i in range(bs):
            seq=x[i].tolist()
            if is_balanced(seq):
                if cp<n_pos:X.append(x[i]);Y.append(1.0);cp+=1
            else:
                if cn<n_neg:X.append(x[i]);Y.append(0.0);cn+=1
        if cp<n_pos:
            k=l//2;seq=[];ore,cre,bal=k,k,0
            while len(seq)<l:
                if ore>0 and(bal==0 or random.random()>0.5 or cre==0):seq.append(0);ore-=1;bal+=1
                elif cre>0 and bal>0:seq.append(1);cre-=1;bal-=1
                elif ore>0:seq.append(0);ore-=1;bal+=1
                else:seq.append(1)
            if is_balanced(seq) and cp<n_pos:X.append(torch.tensor(seq).float());Y.append(1.0);cp+=1
    idx=list(range(n));random.shuffle(idx);X=torch.stack(X)[idx];Y=torch.tensor(Y)[idx].unsqueeze(1)
    return X,Y

# --- Sequential ROUND Model (The Test Subject) ---
class SequentialROUNDModel(nn.Module):
    def __init__(self, hidden_size=64, input_dim=1): # input_dim is per-token (1 for brackets)
        super().__init__()
        self.h = hidden_size
        self.e = nn.Linear(input_dim, hidden_size)
        self.c = PhaseAccumulator(hidden_size)
        # Readout sees standard cos, sin, ph
        self.r = nn.Linear(hidden_size*3, 1)

    def forward(self, x, steps=None): # steps ignored, driven by x length
        # x shape: [Batch, Seq_Len]
        # We treat each element as a time step
        batch_size, seq_len = x.size()
        
        ph = torch.zeros(batch_size, self.h, device=x.device)
        H = []
        
        for t in range(seq_len):
            # Extract current token: [Batch, 1]
            xt = x[:, t].unsqueeze(1)
            
            # Project token to phase features
            p = self.e(xt)
            xp = torch.stack([torch.cos(p), torch.sin(p)], 2)
            
            # Update Phase Dynamics
            ph = self.c(ph, xp)
            H.append(ph)
            
        # Readout from final state
        return self.r(torch.cat([torch.cos(ph), torch.sin(ph), ph], 1)), H

# --- Models ---
class GRUModel(nn.Module):
    def __init__(self,i,h,o=1):super().__init__();self.gru=nn.GRU(i,h,batch_first=True);self.fc=nn.Linear(h,o)
    def forward(self,x):_,h=self.gru(x.unsqueeze(-1));return self.fc(h[-1])

# --- Training Logic ---
def train_round_seq(rid,X,Y,d):
    # Note: input_dim=1 here because we feed tokens sequentially
    m=SequentialROUNDModel(hidden_size=C['hidden_size'],input_dim=1).to(d)
    
    # Using Harmonic Loss
    c=HarmonicROUNDLoss(locking_strength=0.03125,harmonics=[1,2],weights=[1,2],mode='binary',terminal_only=True,floor_clamp=0.032)
    o=optim.Adam(m.parameters(),lr=C['lr'])
    ah=[]
    for e in range(C['epochs']):
        o.zero_grad()
        out,hist=m(X) # X is [Batch, 20]
        l,mse,lok=c(out,Y,hist)
        l.backward()
        o.step()
        preds=(torch.sigmoid(out)>0.5).float()
        acc=(preds==Y).float().mean().item()
        ah.append(acc)
        if e%100==0:P(f"SeqR{rid} E{e}: L={l.item():.4f}, A={acc:.2f}")
    return ah,preds,Y

def train_gru(rid,X,Y,d):
    m=GRUModel(1,C['hidden_size']).to(d);c=nn.BCEWithLogitsLoss();o=optim.Adam(m.parameters(),lr=C['lr']);ah=[]
    for e in range(C['epochs']):
        o.zero_grad();out=m(X);l=c(out,Y);l.backward();o.step()
        preds=(torch.sigmoid(out)>0.5).float();acc=(preds==Y).float().mean().item();ah.append(acc)
        if e%100==0:P(f"G{rid} E{e}: L={l.item():.4f}, A={acc:.2f}")
    return ah,preds,Y

if __name__=="__main__":
    d=torch.device(C['device']);P(f"Dev: {d}")
    X,Y=generate_dyck_data(C['dataset_size'],C['input_dim']);X,Y=X.to(d),Y.to(d)
    
    rr,gr,ap,ft=[],[],[],Y.cpu().numpy()
    
    P("\nTraining Sequential ROUND (Masked/Streaming)")
    for i in range(C['runs']):
        a,p,_=train_round_seq(i+1,X,Y,d)
        rr.append(a)
        ap.append(p.detach().cpu().numpy().flatten())
        
    P("\nTraining GRU")
    for i in range(C['runs']):
        a,p,_=train_gru(i+1,X,Y,d)
        gr.append(a)
        
    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    rm, rs = np.mean(rr, 0), np.std(rr, 0)
    gm, gs = np.mean(gr, 0), np.std(gr, 0)
    ep = np.arange(C['epochs'])

    ax.set_title(f"Sequential ROUND vs GRU: Masked Brackets Test\nstrength=0.03125, harmonics=[1,2], floor=0.032", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy', fontsize=12, color='gray')
    
    max_acc = max(np.max(rm), np.max(gm))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.1)

    ax.fill_between(ep, rm-rs, rm+rs, color='#FF4B4B', alpha=0.1)
    ax.fill_between(ep, gm-gs, gm+gs, color='#4B4BFF', alpha=0.1)
    
    for r in rr: ax.plot(r, color='#FF4B4B', alpha=0.15, linewidth=1)
    for r in gr: ax.plot(r, color='#4B4BFF', alpha=0.15, linewidth=1)
    
    ax.plot(rm, color='#FF4B4B', linewidth=2.5, label='Sequential ROUND')
    ax.plot(gm, color='#4B4BFF', linewidth=2.5, label='GRU (Standard)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'data/benchmark_brackets_masked_{UID}.png', dpi=300)
    P("Done.")
    L_FILE.close()
