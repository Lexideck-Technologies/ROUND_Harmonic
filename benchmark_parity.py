# version 0.6.0 - Harmonic Monism (Parity)
import torch,torch.nn as nn,torch.optim as optim,numpy as np,matplotlib.pyplot as plt,os,uuid
from ROUND import SequentialROUNDModel,HarmonicROUNDLoss
from config import PARITY_CONFIG, get_lock_strength

if not os.path.exists('data'):os.makedirs('data')
UID=os.environ.get('ROUND_BATCH_UID',str(uuid.uuid4())[:8])
output_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
if not os.path.exists(output_dir): os.makedirs(output_dir)
L_FILE=open(f'{output_dir}/log_parity_{UID}.txt','w')
def P(s):print(s);L_FILE.write(str(s)+'\n');L_FILE.flush()

P(f"Batch UID: {UID}")
# Load Config
TC = PARITY_CONFIG
P(f"Run Config: {TC}")
C={'task':'parity_16','input_dim':16,'hidden_size':1,'steps':20,'epochs':TC['EPOCHS'],'batch_size':64,'dataset_size':2000,'runs':5,'lr':TC['LR'],'device':'cuda' if torch.cuda.is_available() else 'cpu'}

def generate_parity_data(n,b):X=torch.randint(0,2,(n,b)).float();return X,(X.sum(1)%2).unsqueeze(1).float()

class GRUModel(nn.Module):
    def __init__(self,i,h,o=1):super().__init__();self.gru=nn.GRU(i,h,batch_first=True);self.fc=nn.Linear(h,o)
    def forward(self,x):_,h=self.gru(x.unsqueeze(-1));return self.fc(h[-1])

def train_round(rid,X,Y,Xt,Yt,d):
    # Sequential Parity: Input Dim = 1 (Bit stream), Seq Len = 16
    m=SequentialROUNDModel(hidden_size=C['hidden_size'],input_dim=1,output_classes=1,spinor=True,use_raw_phase=False).to(d)
    
    # Initialize to break symmetry
    with torch.no_grad():
         m.e.weight.fill_(3.14)
         m.e.bias.fill_(0.0)
         # Init accumulator to respond to input
         m.c.d.bias.fill_(1.57) # PI/2
         m.c.d.weight.zero_()
         m.c.d.weight[:, 4].fill_(-1.57) # InCos index
         
    c=HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                        harmonics=TC['HARMONICS'],
                        weights=[1.0]*len(TC['HARMONICS']),
                        mode='binary',
                        terminal_only=False)
    o=optim.Adam(m.parameters(),lr=0.01) # Increased LR for Parity
    ah=[];locked=False
    for e in range(C['epochs']):
        if e == (C['epochs'] // 2):
            for g in o.param_groups: g['lr'] *= 0.1

        # Delayed Locking: Allow free exploration first
        delay_threshold = TC.get('DELAYED_LOCKING', 0.4) * C['epochs']
        if e < delay_threshold:
             c.locking_strength = 0.0
        else:
             c.locking_strength = get_lock_strength(e, C['epochs'], TC['PEAK_LOCKING_STRENGTH'])
        o.zero_grad()
        # X: [Batch, 16] -> [Batch, 16, 1]
        out,h=m(X.unsqueeze(-1))
        
        # Loss & Step
        l,tk,lk=c(out,Y,h)
        l.backward()
        o.step()
        
        # Test Accuracy
        with torch.no_grad():
            out_t, _ = m(Xt.unsqueeze(-1))
            pt = (torch.sigmoid(out_t)>0.5).float()
            acc_t = (pt == Yt).float().mean().item()
            
        ah.append(acc_t)
        
        if e%100==0 or e == C['epochs'] - 1:P(f"R{rid} E{e}: TestAcc={acc_t:.2f} | Lock={lk:.4f}")
        
    return ah,pt,Yt

def train_gru(rid,X,Y,Xt,Yt,d):
    m=GRUModel(1,C['hidden_size']).to(d);c=nn.BCEWithLogitsLoss();o=optim.Adam(m.parameters(),lr=C['lr']);ah=[];locked=False
    for e in range(C['epochs']):
        o.zero_grad();out=m(X);l=c(out,Y);l.backward();o.step()
        
        # Test Accuracy
        with torch.no_grad():
            out_t = m(Xt)
            pt = (torch.sigmoid(out_t)>0.5).float()
            acc_t = (pt == Yt).float().mean().item()
            
        ah.append(acc_t)
        if e % 100 == 0 or e == C['epochs'] - 1:P(f"G{rid} E{e}: L={l.item():.4f}, TestAcc={acc_t:.2f}")
    return ah,pt,Yt

if __name__=="__main__":
    d=torch.device(C['device']);P(f"Dev: {d}")
    # Train Split
    X,Y=generate_parity_data(C['dataset_size'],16);X,Y=X.to(d),Y.to(d) # Hardcoded 16 dim from config
    # Test Split (Validation)
    Xt,Yt=generate_parity_data(1000,16);Xt,Yt=Xt.to(d),Yt.to(d)
    
    rr,gr,ap,ft=[],[],[],Yt.cpu().numpy();P("Training ROUND (Test Validation)")
    for i in range(C['runs']):a,p,_=train_round(i+1,X,Y,Xt,Yt,d);rr.append(a);ap.append(p.detach().cpu().numpy().flatten())
    P("Training GRU (Test Validation)")
    for i in range(C['runs']):a,p,_=train_gru(i+1,X,Y,Xt,Yt,d);gr.append(a)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    rm, rs = np.mean(rr, 0), np.std(rr, 0)
    gm, gs = np.mean(gr, 0), np.std(gr, 0)
    ep = np.arange(C['epochs'])

    ax.set_title(f"Harmonic ROUND vs GRU: 16-bit Parity (Harmonic Monism)\nstrength={TC['PEAK_LOCKING_STRENGTH']}, harmonics={TC['HARMONICS']}", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy', fontsize=12, color='gray')
    
    max_acc = max(np.max(rm), np.max(gm))
    if max_acc < 0.95:
        ax.set_ylim(-0.05, min(1.05, max_acc + 0.1))
    else:
        ax.set_ylim(-0.05, 1.05)

    ax.grid(True, alpha=0.1)

    ax.fill_between(ep, rm-rs, rm+rs, color='#FF4B4B', alpha=0.1)
    ax.fill_between(ep, gm-gs, gm+gs, color='#4B4BFF', alpha=0.1)
    
    for r in rr: ax.plot(r, color='#FF4B4B', alpha=0.15, linewidth=1)
    for r in gr: ax.plot(r, color='#4B4BFF', alpha=0.15, linewidth=1)
    
    ax.plot(rm, color='#FF4B4B', linewidth=2.5, label='ROUND (Monism)')
    ax.plot(gm, color='#4B4BFF', linewidth=2.5, label='GRU (Standard)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'benchmark_parity_{UID}.png'), dpi=300)
    
    # Correlation Plot
    ds = np.vstack([np.stack(ap), ft.flatten()])
    corr = np.corrcoef(ds)
    labels = [f'R{i+1}' for i in range(C['runs'])] + ['GT']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'ROUND Consistency: Parity\nBatch {UID}')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if 0.3 < corr[i, j] < 0.7 else "white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_parity_{UID}.png'), dpi=300)
    P("Done.")
    L_FILE.close()
