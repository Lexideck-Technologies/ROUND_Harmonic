# version 0.7.3 - "The Hyper-Resolution Basin" (Masked Brackets)
import torch,torch.nn as nn,torch.optim as optim,numpy as np,matplotlib.pyplot as plt,os,uuid
from ROUND import SequentialROUNDModel,HarmonicROUNDLoss
from config import BRACKETS_CONFIG, get_lock_strength

if not os.path.exists('data'):os.makedirs('data')
UID=os.environ.get('ROUND_BATCH_UID',str(uuid.uuid4())[:8])
base_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
output_dir = os.path.join(base_dir, UID)
if not os.path.exists(output_dir): os.makedirs(output_dir)
L_FILE=open(f'{output_dir}/log_brackets_masked_{UID}.txt','w')
def P(s):print(s);L_FILE.write(str(s)+'\n');L_FILE.flush()

P(f"Batch UID: {UID}")
# Load Config
TC = BRACKETS_CONFIG
P(f"Run Config: {TC}")
C={'task':'dyck_2','seq_len':20,'hidden_r':TC['HIDDEN_R'],'hidden_g':TC['HIDDEN_G'],'steps':20,'epochs':TC['EPOCHS'],
   'batch_size':64,'dataset_size':2000,'runs':5,'lr':TC['LR'],'device':'cuda' if torch.cuda.is_available() else 'cpu'}

def generate_dyck_data(n, l):
    X, Y = [], []
    def gen_valid(length):
        if length == 0: return []
        s = []
        bal = 0
        for i in range(length):
            rem = length - 1 - i
            opts = []
            if bal < rem: opts.append(1.0)
            if bal > 0: opts.append(-1.0)
            if not opts: return [1.0]*length
            c = opts[np.random.randint(len(opts))]
            s.append(c)
            bal += int(c)
        return s
    
    for i in range(n):
        if i % 2 == 0: # Valid
            s = gen_valid(l)
            Y.append(1.0)
        else: # Invalid
            s = gen_valid(l)
            idx = np.random.randint(l)
            s[idx] *= -1
            bal = 0; v = True
            for x in s:
                bal += x
                if bal < 0: v = False
            if bal != 0: v = False
            Y.append(1.0 if v else 0.0)
        X.append(torch.tensor(s).float())
    return torch.stack(X), torch.tensor(Y).unsqueeze(1).float()

class GRUModel(nn.Module):
    def __init__(self,i,h,o=1):super().__init__();self.gru=nn.GRU(i,h,batch_first=True);self.fc=nn.Linear(h,o)
    def forward(self,x):_,h=self.gru(x.unsqueeze(-1));return self.fc(h[-1])

def train_round_seq(rid,X,Y,Xt,Yt,d):
    # Streaming Input: Input Dim = 1
    m=SequentialROUNDModel(hidden_size=TC['HIDDEN_R'],input_dim=1).to(d)
    c=HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                        harmonics=TC['HARMONICS'],
                        weights=[1.0]*len(TC['HARMONICS']),
                        mode='binary',
                        terminal_only=TC.get('TERMINAL_ONLY', True))
    o=optim.Adam(m.parameters(),lr=TC['LR'])
    ah=[];locked=False
    for e in range(TC['EPOCHS']):
        # Delayed Locking: Open up learning curve to 50%
        delay_threshold = 0.5 * TC['EPOCHS']
        if e < delay_threshold:
            c.locking_strength = 0.0
        else:
            c.locking_strength = get_lock_strength(e, TC['EPOCHS'], TC['PEAK_LOCKING_STRENGTH'], TC.get('FLOOR', 0.015625))
        o.zero_grad()
        out,h=m(X.unsqueeze(-1))
        l,tk,lk=c(out,Y,h)
        l.backward()
        o.step()
        
        with torch.no_grad():
            out_t, _ = m(Xt.unsqueeze(-1))
            pt = (torch.sigmoid(out_t)>0.5).float()
            acc_t = (pt == Yt).float().mean().item()
            
        ah.append(acc_t)
        if e % 100 == 0 or e == TC['EPOCHS'] - 1:
            P(f"SeqR{rid} E{e}: A={acc_t:.2f} | K={lk:.4f}")
        
    return ah,pt,Yt

def train_gru(rid,X,Y,Xt,Yt,d):
    m=GRUModel(1,C['hidden_g']).to(d);c=nn.BCEWithLogitsLoss();o=optim.Adam(m.parameters(),lr=C['lr']);ah=[];locked=False
    for e in range(C['epochs']):
        o.zero_grad();out=m(X);l=c(out,Y);l.backward();o.step()
        with torch.no_grad():
            out_t = m(Xt)
            pt = (torch.sigmoid(out_t)>0.5).float()
            acc_t = (pt == Yt).float().mean().item()
        ah.append(acc_t)
        if e % 100 == 0 or e == TC['EPOCHS'] - 1:
            P(f"G{rid} E{e}: L={l.item():.4f}, TestAcc={acc_t:.2f}")
    return ah,pt,Yt

if __name__=="__main__":
    d=torch.device(C['device']);P(f"Dev: {d}")
    X,Y=generate_dyck_data(C['dataset_size'],C['seq_len']);X,Y=X.to(d),Y.to(d)
    Xt,Yt=generate_dyck_data(1000,C['seq_len']);Xt,Yt=Xt.to(d),Yt.to(d)
    
    rr,gr,ap,ft=[],[],[],Yt.cpu().numpy();P("Training Sequential ROUND (Test Validation)")
    for i in range(C['runs']):a,p,_=train_round_seq(i+1,X,Y,Xt,Yt,d);rr.append(a);ap.append(p.detach().cpu().numpy().flatten())
    P("Training GRU (Test Validation)")
    for i in range(C['runs']):a,p,_=train_gru(i+1,X,Y,Xt,Yt,d);gr.append(a)
    
    # Plotting with Seaborn
    from visualization_utils import setup_seaborn_theme, prepare_comparison_data, plot_benchmark_comparison

    palette = setup_seaborn_theme(style='darkgrid', palette='classic')
    df = prepare_comparison_data(rr, gr)

    title = f"Harmonic ROUND vs GRU: Masked Brackets [ROUND={TC['HIDDEN_R']} Neurons, GRU={TC['HIDDEN_G']} Neurons]\nstrength={TC['PEAK_LOCKING_STRENGTH']}, harmonics={TC['HARMONICS']}"
    plot_benchmark_comparison(
        df=df,
        title=title,
        palette=palette,
        output_path=os.path.join(output_dir, f'benchmark_brackets_masked_{UID}.png'),
        figsize=(12, 6)
    )
    
    L_FILE.close()
