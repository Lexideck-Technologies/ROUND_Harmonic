# version 0.7.3 - "The Hyper-Resolution Basin" (Topology)
import torch,torch.nn as nn,torch.optim as optim,numpy as np,matplotlib.pyplot as plt,os,uuid
from ROUND import SequentialROUNDModel,HarmonicROUNDLoss
from config import TOPOLOGY_CONFIG, get_lock_strength

if not os.path.exists('data'):os.makedirs('data')
UID=os.environ.get('ROUND_BATCH_UID',str(uuid.uuid4())[:8])
base_dir = os.environ.get('ROUND_OUTPUT_DIR', 'data')
output_dir = os.path.join(base_dir, UID)
if not os.path.exists(output_dir): os.makedirs(output_dir)
L_FILE=open(f'{output_dir}/log_topology_{UID}.txt','w')
def P(s):print(s);L_FILE.write(str(s)+'\n');L_FILE.flush()

P(f"Batch UID: {UID}")
# Load Config
TC = TOPOLOGY_CONFIG
P(f"Run Config: {TC}")
C={'task':'euler_char',
   'seq_len':100, # Matches generated flatten size
   'hidden_r':TC['HIDDEN_R'],
   'hidden_g':TC['HIDDEN_G'],
   'epochs':TC['EPOCHS'],
   'runs':5, # Standard
   'lr':TC['LR'],
   'dataset_size':2000,
   'device':'cuda' if torch.cuda.is_available() else 'cpu'}

def generate_topology_data(n,seq_len):
    # Generates Flattened Adjacency Matrices for Graphs (Nodes=10, Len=100)
    # Class 0: Tree (No Cycles) | Class 1: Cyclic (Cycle present)
    X=torch.zeros(n,seq_len)
    Y=torch.zeros(n,1)
    
    for i in range(n):
        adj=np.zeros((10,10))
        # Build Tree
        nodes=list(range(10))
        np.random.shuffle(nodes)
        visited=[nodes[0]]
        unvisited=nodes[1:]
        while unvisited:
            u=np.random.choice(visited)
            v=unvisited.pop()
            adj[u,v]=adj[v,u]=1
            visited.append(v)
        
        label=0
        if np.random.rand()>0.5:
            label=1
            n_extra=np.random.randint(1,4)
            added = 0
            while added < n_extra:
                u,v=np.random.choice(10,2,replace=False)
                if adj[u,v] == 0:
                    adj[u,v]=adj[v,u]=1
                    added += 1
        Y[i]=label
        X[i] = torch.tensor(adj.flatten()).float()
    return X,Y

class GRUModel(nn.Module):
    def __init__(self,i,h,o=1):super().__init__();self.gru=nn.GRU(i,h,batch_first=True);self.fc=nn.Linear(h,o)
    def forward(self,x):_,h=self.gru(x);return self.fc(h[-1])

def train_round(rid,X,Y,Xt,Yt,d):
    m=SequentialROUNDModel(hidden_size=TC['HIDDEN_R'],input_dim=1,output_classes=1,wobble=TC['WOBBLE']).to(d)
    
    # HARMONICS TEST CONFIGURATION
    c=HarmonicROUNDLoss(locking_strength=TC['PEAK_LOCKING_STRENGTH'],
                        harmonics=TC['HARMONICS'],
                        weights=[1.0]*len(TC['HARMONICS']), 
                        mode='binary',
                        terminal_only=True)
                        
    o=optim.Adam(m.parameters(),lr=TC['LR'])
    ah=[];locked=False
    for e in range(TC['EPOCHS']):
        # Delayed Locking: Open up learning curve to 50%
        delay_threshold = 0.5 * TC['EPOCHS']
        if e < delay_threshold:
             c.locking_strength = 0.0
        else:
             c.locking_strength = get_lock_strength(e, TC['EPOCHS'], TC['PEAK_LOCKING_STRENGTH'], TC['FLOOR'])
        
        o.zero_grad()
        # Input Dim 1. X is [Batch, 30] so unsqueeze to [Batch, 30, 1]
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
            P(f"R{rid} E{e}: TestAcc={acc_t:.2f} | Lock={lk:.4f}")
    return ah,pt,Yt

def train_gru(rid,X,Y,Xt,Yt,d):
    m=GRUModel(1,C['hidden_g']).to(d);c=nn.BCEWithLogitsLoss();o=optim.Adam(m.parameters(),lr=C['lr']);ah=[];locked=False
    for e in range(C['epochs']):
        o.zero_grad();out=m(X.unsqueeze(-1));l=c(out,Y);l.backward();o.step()
        with torch.no_grad():
            out_t = m(Xt.unsqueeze(-1))
            pt = (torch.sigmoid(out_t)>0.5).float()
            acc_t = (pt == Yt).float().mean().item()
        ah.append(acc_t)
        if e % 100 == 0 or e == TC['EPOCHS'] - 1:
            P(f"G{rid} E{e}: L={l.item():.4f}, TestAcc={acc_t:.2f}")
    return ah,pt,Yt

if __name__=="__main__":
    d=torch.device(C['device']);P(f"Dev: {d}")
    # Train Split
    X,Y=generate_topology_data(C['dataset_size'],C['seq_len']);X,Y=X.to(d),Y.to(d)
    
    # FILTER ZEROS STRATEGY (SORTING)
    if TC.get('SORT_INPUTS', False):
        X, _ = torch.sort(X, descending=True, dim=1)
        # Truncate to Max Edges + Buffer (30)
        X = X[:, :30]
        
    # Test Split
    Xt,Yt=generate_topology_data(1000,C['seq_len']);Xt,Yt=Xt.to(d),Yt.to(d)
    if TC.get('SORT_INPUTS', False):
        Xt, _ = torch.sort(Xt, descending=True, dim=1)
        Xt = Xt[:, :30]
        
    rr,gr,ap,ft=[],[],[],Yt.cpu().numpy();P("Training ROUND (Test Validation)")
    for i in range(C['runs']):a,p,_=train_round(i+1,X,Y,Xt,Yt,d);rr.append(a);ap.append(p.detach().cpu().numpy().flatten())
    P("Training GRU (Test Validation)")
    for i in range(C['runs']):a,p,_=train_gru(i+1,X,Y,Xt,Yt,d);gr.append(a)
    
    # Plotting with Seaborn (with individual runs)
    from visualization_utils import setup_seaborn_theme, prepare_comparison_data, plot_benchmark_with_runs

    palette = setup_seaborn_theme(style='darkgrid', palette='classic')
    df = prepare_comparison_data(rr, gr)

    title = f"Harmonic ROUND vs GRU: Graph Cycle [ROUND={TC['HIDDEN_R']} Neurons, GRU={TC['HIDDEN_G']} Neurons]\nstrength={TC['PEAK_LOCKING_STRENGTH']}, harmonics={TC['HARMONICS']}"

    # Calculate ylim
    max_acc = df['Accuracy'].max()
    if max_acc < 0.95:
        ylim = (-0.05, min(1.05, max_acc + 0.1))
    else:
        ylim = (-0.05, 1.05)

    plot_benchmark_with_runs(
        df=df,
        title=title,
        palette=palette,
        output_path=os.path.join(output_dir, f'benchmark_topology_{UID}.png'),
        ylim=ylim
    )
    
    P("Done.")
    L_FILE.close()
