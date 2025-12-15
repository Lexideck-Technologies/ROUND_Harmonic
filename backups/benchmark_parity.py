
# version 0.3.1
import torch,torch.nn as nn,torch.optim as optim,numpy as np,matplotlib.pyplot as plt,os,uuid
from ROUND import ROUNDModel,ROUNDLoss,HarmonicROUNDLoss
if not os.path.exists('data'):os.makedirs('data')
UID=os.environ.get('ROUND_BATCH_UID',str(uuid.uuid4())[:8])
L_FILE=open(f'data/log_parity_{UID}.txt','w')
def P(s):print(s);L_FILE.write(str(s)+'\n');L_FILE.flush()
P(f"Batch UID: {UID}")
C={'task':'parity_16','input_dim':16,'hidden_size':32,'steps':20,'epochs':1000,'batch_size':64,'dataset_size':2000,'runs':5,'lr':0.001953125,'device':'cuda' if torch.cuda.is_available() else 'cpu'}
def generate_parity_data(n,b):X=torch.randint(0,2,(n,b)).float();return X,(X.sum(1)%2).unsqueeze(1).float()
class GRUModel(nn.Module):
    def __init__(self,i,h,o=1):super().__init__();self.gru=nn.GRU(i,h,batch_first=True);self.fc=nn.Linear(h,o)
    def forward(self,x):_,h=self.gru(x.unsqueeze(-1));return self.fc(h[-1])
def train_round(rid,X,Y,d):
    m=ROUNDModel(hidden_size=C['hidden_size'],input_dim=C['input_dim']).to(d)
    c=HarmonicROUNDLoss(locking_strength=0.03125,harmonics=[1,2],weights=[1,2],mode='binary',terminal_only=True,floor_clamp=0.032);o=optim.Adam(m.parameters(),lr=C['lr']);ah=[]
    for e in range(C['epochs']):
        o.zero_grad();out,h=m(X,steps=C['steps']);l,_,_=c(out,Y,h);l.backward();o.step()
        preds=(torch.sigmoid(out)>0.5).float();acc=(preds==Y).float().mean().item();ah.append(acc)
        if e%100==0:P(f"R{rid} E{e}: L={l.item():.4f}, A={acc:.2f}")
    return ah,preds,Y
def train_gru(rid,X,Y,d):
    m=GRUModel(1,C['hidden_size']).to(d);c=nn.BCEWithLogitsLoss();o=optim.Adam(m.parameters(),lr=C['lr']);ah=[]
    for e in range(C['epochs']):
        o.zero_grad();out=m(X);l=c(out,Y);l.backward();o.step()
        preds=(torch.sigmoid(out)>0.5).float();acc=(preds==Y).float().mean().item();ah.append(acc)
        if e%100==0:P(f"G{rid} E{e}: L={l.item():.4f}, A={acc:.2f}")
    return ah,preds,Y
if __name__=="__main__":
    d=torch.device(C['device']);P(f"Dev: {d}");X,Y=generate_parity_data(C['dataset_size'],C['input_dim']);X,Y=X.to(d),Y.to(d)
    rr,gr,ap,ft=[],[],[],Y.cpu().numpy();P("Training ROUND")
    for i in range(C['runs']):a,p,_=train_round(i+1,X,Y,d);rr.append(a);ap.append(p.detach().cpu().numpy().flatten())
    P("Training GRU")
    for i in range(C['runs']):a,p,_=train_gru(i+1,X,Y,d);gr.append(a)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    rm, rs = np.mean(rr, 0), np.std(rr, 0)
    gm, gs = np.mean(gr, 0), np.std(gr, 0)
    ep = np.arange(C['epochs'])

    ax.set_title(f"Harmonic ROUND vs GRU: 16-bit Parity (XOR)\nstrength=0.03125, harmonics=[1,2], floor=0.032", fontsize=14, color='white')
    ax.set_xlabel('Epochs', fontsize=12, color='gray')
    ax.set_ylabel('Accuracy', fontsize=12, color='gray')
    
    # Dynamic Scaling for Failure Mode Analysis
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
    
    ax.plot(rm, color='#FF4B4B', linewidth=2.5, label='ROUND (Harmonic)')
    ax.plot(gm, color='#4B4BFF', linewidth=2.5, label='GRU (Standard)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'data/benchmark_parity_{UID}.png', dpi=300)

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
    plt.savefig(f'data/correlation_parity_{UID}.png', dpi=300)
    P("Done.")
    L_FILE.close()
