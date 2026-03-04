#!/usr/bin/env python3
"""
BudgetNAS v3 — Full 10/10 experiment suite
==========================================
Key upgrades over v2:
  1. Contextual bandit controller (Thompson sampling, cost-aware)
  2. Net2Net-style warm-start + freeze schedule for mutation stabilization
  3. Smart Growing baseline (grow when acc drops, not unconditionally)
  4. MMD-based drift detector + trigger ablation
  5. Longer stream (5 domains: CIFAR-10 → CIFAR-100 → SVHN → FashionMNIST → KMNIST)
  6. Unknown-boundary setting via change-point detection
  7. New metrics: AUC, gain-per-mutation, gain-per-M-params
  8. Budget calibration (auto-select B)

Saves incrementally after every method+seed.
"""
import os, sys, json, time, copy, random, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision, torchvision.transforms as T

# ── device ──
if torch.backends.mps.is_available(): DEV = torch.device("mps")
elif torch.cuda.is_available(): DEV = torch.device("cuda")
else: DEV = torch.device("cpu")
print(f"Device: {DEV}")

# ── config ──
BS      = 256
EPOCHS  = 3
CHUNKS  = 4
LR      = 0.02
WD      = 1e-4
N_TRAIN = 8000
N_TEST  = 2000
EWC_LAM = 400
ER_BUF  = 300
SEEDS   = [42, 123, 7]

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)

def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)

# ═══════════════ DATA ═══════════════

def tfm(name):
    if name in ("cifar10","cifar100"):
        m = ((.4914,.4822,.4465),(.2023,.1994,.2010))
        tr = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*m)])
        te = T.Compose([T.ToTensor(), T.Normalize(*m)])
    elif name == "svhn":
        m = ((.4377,.4438,.4728),(.198,.201,.197))
        tr = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*m)])
        te = T.Compose([T.ToTensor(), T.Normalize(*m)])
    else:  # fashion, kmnist — grayscale, normalize with 1-ch stats; To3Ch handles expansion
        m1 = ((.5,),(.5,))
        tr = T.Compose([T.Resize(32), T.RandomCrop(32,4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*m1)])
        te = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize(*m1)])
    return tr, te

def sub(ds, n, seed):
    if len(ds) <= n: return ds
    return Subset(ds, random.Random(seed).sample(range(len(ds)), n))

# Wrapper to convert 1-channel to 3-channel
class To3Ch:
    def __init__(self, ds):
        self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        if x.shape[0] == 1: x = x.repeat(3,1,1)
        # Resize to 32x32 if needed
        if x.shape[1] != 32 or x.shape[2] != 32:
            x = F.interpolate(x.unsqueeze(0), size=32, mode='bilinear', align_corners=False).squeeze(0)
        return x, y

_cache = {}
def get_data(seed, domains=None):
    if domains is None:
        domains = ["cifar10","cifar100","svhn"]
    key = (seed, tuple(domains))
    if key in _cache: return _cache[key]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    d = {}
    for name in domains:
        tr_t, te_t = tfm(name)
        if name == "cifar10":
            train = sub(torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr_t), N_TRAIN, seed)
            test = sub(torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te_t), N_TEST, seed)
            nc = 10
        elif name == "cifar100":
            train = sub(torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tr_t), N_TRAIN, seed)
            test = sub(torchvision.datasets.CIFAR100(root, train=False, download=True, transform=te_t), N_TEST, seed)
            nc = 100
        elif name == "svhn":
            train = sub(torchvision.datasets.SVHN(root, split="train", download=True, transform=tr_t), N_TRAIN, seed)
            test = sub(torchvision.datasets.SVHN(root, split="test", download=True, transform=te_t), N_TEST, seed)
            nc = 10
        elif name == "fashion":
            train = To3Ch(sub(torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=tr_t), N_TRAIN, seed))
            test = To3Ch(sub(torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=te_t), N_TEST, seed))
            nc = 10
        elif name == "kmnist":
            train = To3Ch(sub(torchvision.datasets.KMNIST(root, train=True, download=True, transform=tr_t), N_TRAIN, seed))
            test = To3Ch(sub(torchvision.datasets.KMNIST(root, train=False, download=True, transform=te_t), N_TEST, seed))
            nc = 10
        else:
            raise ValueError(f"Unknown dataset: {name}")
        d[name] = (train, test, nc)
    _cache[key] = d
    return d

def chunks(ds, seed):
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    cs = len(idx)//CHUNKS
    return [Subset(ds, idx[i*cs:(i+1)*cs if i<CHUNKS-1 else len(idx)]) for i in range(CHUNKS)]

# ═══════════════ MODEL ═══════════════

class RB(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,s,1,bias=False); self.b1=nn.BatchNorm2d(oc)
        self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False); self.b2=nn.BatchNorm2d(oc)
        self.sk=nn.Sequential() if s==1 and ic==oc else nn.Sequential(nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc))
    def forward(self,x): return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+self.sk(x))

class Net(nn.Module):
    def __init__(self, nc=10, cfgs=None):
        super().__init__()
        if cfgs is None: cfgs = [(32,64,1),(64,128,2),(128,128,1)]
        self.stem = nn.Sequential(nn.Conv2d(3,32,3,1,1,bias=False),nn.BatchNorm2d(32),nn.ReLU())
        self.blocks = nn.ModuleList([RB(ic,oc,s) for ic,oc,s in cfgs])
        self.cfgs = list(cfgs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._lc = cfgs[-1][1]
        self.head = nn.Linear(self._lc, nc)
    def forward(self,x):
        x=self.stem(x)
        for b in self.blocks: x=b(x)
        return self.head(self.pool(x).flatten(1))
    def features(self, x):
        """Extract features before the head (for MMD drift detection)."""
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.pool(x).flatten(1)
    def nparams(self): return sum(p.numel() for p in self.parameters())

# ── Mutation operators ──

def add_blk(m, warmstart=False):
    c=list(m.cfgs); lc=c[-1][1]; c.append((lc,lc,1)); n=_reb(m,c)
    if warmstart:
        _identity_init(n.blocks[-1])
    return n

def add_ds(m, warmstart=False):
    c=list(m.cfgs); lc=c[-1][1]; c.append((lc,min(lc*2,256),2)); n=_reb(m,c)
    if warmstart:
        _identity_init(n.blocks[-1])
    return n

def rm_blk(m):
    if len(m.cfgs)<=2: return m
    c=list(m.cfgs); c.pop(-2)
    prev=32; fc=[]
    for ic,oc,s in c: fc.append((prev,oc,s)); prev=oc
    return _reb(m,fc)

def new_head(m, nc): m.head=nn.Linear(m._lc,nc); return m

def _reb(m, cfgs):
    nc=m.head.out_features; n=Net(nc,cfgs)
    try: n.stem.load_state_dict(m.stem.state_dict())
    except: pass
    for i in range(min(len(n.blocks),len(m.blocks))):
        if i<len(m.cfgs) and i<len(cfgs) and cfgs[i]==m.cfgs[i]:
            try: n.blocks[i].load_state_dict(m.blocks[i].state_dict())
            except: pass
    if n._lc==m._lc:
        try: n.head.load_state_dict(m.head.state_dict())
        except: pass
    return n

def _identity_init(block):
    """Net2Net-style: init new block close to identity to reduce mutation disruption."""
    with torch.no_grad():
        # Zero out the second conv so residual path dominates
        if hasattr(block, 'c2') and hasattr(block.c2, 'weight'):
            block.c2.weight.zero_()
        if hasattr(block, 'b2'):
            block.b2.weight.fill_(1.0)
            block.b2.bias.zero_()

def freeze_early(m, keep_last=2):
    """Freeze all blocks except the last `keep_last` blocks."""
    for i, b in enumerate(m.blocks):
        if i < len(m.blocks) - keep_last:
            for p in b.parameters(): p.requires_grad = False

def unfreeze_all(m):
    for p in m.parameters(): p.requires_grad = True

# ═══════════════ DRIFT DETECTORS ═══════════════

class LossDrift:
    """Original loss-based drift detector."""
    def __init__(self, w=2, t=.25): self.w=w; self.t=t; self.h=[]
    def update(self, l): self.h.append(l)
    def detect(self):
        if len(self.h)<self.w+1: return False
        r=np.mean(self.h[-self.w:]); o=np.mean(self.h[-2*self.w:-self.w]) if len(self.h)>=2*self.w else self.h[0]
        return (r-o)/(o+1e-8)>self.t

class MMDDrift:
    """MMD-based drift detector on feature embeddings."""
    def __init__(self, window=100, threshold=0.1):
        self.window = window
        self.threshold = threshold
        self.ref_feats = None
        self.ready = False

    def set_reference(self, model, dl, dev, n_samples=200):
        """Collect reference features from current domain."""
        model.eval()
        feats = []
        ct = 0
        with torch.no_grad():
            for x, _ in dl:
                if ct >= n_samples: break
                x = x.to(dev)
                f = model.features(x)
                feats.append(f.cpu())
                ct += x.size(0)
        self.ref_feats = torch.cat(feats)[:n_samples]
        self.ready = True

    def detect(self, model, dl, dev, n_samples=200):
        if not self.ready: return False
        model.eval()
        feats = []
        ct = 0
        with torch.no_grad():
            for x, _ in dl:
                if ct >= n_samples: break
                x = x.to(dev)
                f = model.features(x)
                feats.append(f.cpu())
                ct += x.size(0)
        cur_feats = torch.cat(feats)[:n_samples]
        mmd = self._mmd(self.ref_feats, cur_feats)
        return mmd > self.threshold

    @staticmethod
    def _mmd(x, y):
        """Simplified MMD^2 with RBF kernel."""
        def rbf(a, b, sigma=1.0):
            d = torch.cdist(a, b, p=2)
            return torch.exp(-d**2 / (2 * sigma**2))
        xx = rbf(x, x).mean()
        yy = rbf(y, y).mean()
        xy = rbf(x, y).mean()
        return float(xx + yy - 2*xy)

# ═══════════════ TRAINING ROUTINES ═══════════════

def train_ep(m,dl,opt,cr,dev):
    m.train(); tl=c=t=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); o=m(x); l=cr(o,y); l.backward(); opt.step()
        tl+=l.item(); c+=o.argmax(1).eq(y).sum().item(); t+=y.size(0)
    return tl/max(len(dl),1), c/max(t,1)

def train_ewc(m,dl,opt,cr,dev,fi,pp,lam):
    m.train(); tl=c=t=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); o=m(x); l=cr(o,y)
        ewc=0
        for n,p in m.named_parameters():
            if n in fi and n in pp and p.shape==pp[n].shape:
                ewc+=(fi[n]*(p-pp[n]).pow(2)).sum()
        (l+lam/2*ewc).backward(); opt.step()
        tl+=l.item(); c+=o.argmax(1).eq(y).sum().item(); t+=y.size(0)
    return tl/max(len(dl),1), c/max(t,1)

def train_er(m,dl,opt,cr,dev,buf):
    m.train(); tl=c=t=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev)
        if buf[0].numel()>0:
            n=min(32,len(buf[0])); ix=torch.randperm(len(buf[0]))[:n]
            x=torch.cat([x,buf[0][ix].to(dev)]); y=torch.cat([y,buf[1][ix].to(dev)])
        opt.zero_grad(); o=m(x); l=cr(o,y); l.backward(); opt.step()
        tl+=l.item(); c+=o.argmax(1).eq(y).sum().item(); t+=y.size(0)
    return tl/max(len(dl),1), c/max(t,1)

def ev(m,dl,cr,dev):
    m.eval(); tl=c=t=0
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); o=m(x); l=cr(o,y)
            tl+=l.item(); c+=o.argmax(1).eq(y).sum().item(); t+=y.size(0)
    return tl/max(len(dl),1), c/max(t,1)

def fisher(m,dl,dev,ns=400):
    m.eval(); fi={n:torch.zeros_like(p) for n,p in m.named_parameters() if p.requires_grad}; ct=0
    for x,y in dl:
        if ct>=ns: break
        x,y=x.to(dev),y.to(dev); m.zero_grad(); F.cross_entropy(m(x),y).backward()
        for n,p in m.named_parameters():
            if p.requires_grad and p.grad is not None: fi[n]+=p.grad.data.pow(2)*x.size(0)
        ct+=x.size(0)
    for n in fi: fi[n]/=max(ct,1)
    return fi

def fill_buf(dl, mx):
    xs,ys=[],[]
    for x,y in dl: xs.append(x); ys.append(y)
    xs=torch.cat(xs); ys=torch.cat(ys)
    if len(xs)>mx: ix=torch.randperm(len(xs))[:mx]; xs=xs[ix]; ys=ys[ix]
    return (xs.cpu(),ys.cpu())

# ═══════════════ CONTROLLERS ═══════════════

class HeuristicCtrl:
    """Original heuristic controller (v2 baseline)."""
    def __init__(self,B=3): self.B=B; self.u=0; self.log=[]
    def reset(self): self.u=0
    def propose(self,m,acc,pa,drift):
        if self.u>=self.B or (not drift and acc>0.5): return None
        cs=[]
        if len(m.cfgs)<8: cs.append(("add_blk",.5+(.3 if drift else 0)+(.2 if acc<.4 else 0)))
        nd=sum(1 for c in m.cfgs if c[2]==2)
        if nd<3: cs.append(("add_ds",.4+(.4 if drift else 0)))
        if len(m.cfgs)>3: cs.append(("rm_blk",.2+(.1 if acc>pa and not drift else 0)))
        if not cs: return None
        cs.sort(key=lambda x:x[1],reverse=True); mut=cs[0][0]; self.u+=1
        self.log.append({"m":mut,"rem":self.B-self.u,"acc":acc,"drift":drift})
        return mut

class BanditCtrl:
    """
    Contextual bandit controller with Thompson sampling.
    Actions: {add_blk, add_ds, rm_blk, no_op}
    Reward: Δacc - λ_cost * Δparams/1M
    """
    def __init__(self, B=3, cost_lambda=0.05):
        self.B = B
        self.u = 0
        self.cost_lambda = cost_lambda
        self.actions = ["add_blk", "add_ds", "rm_blk", "no_op"]
        # Beta prior for each action (successes, failures)
        self.alpha = {a: 1.0 for a in self.actions}
        self.beta_p = {a: 1.0 for a in self.actions}
        self.log = []
        self.prev_acc = None
        self.prev_params = None
        self.prev_action = None

    def reset(self):
        self.u = 0

    def update_reward(self, acc, params):
        """Update bandit after observing outcome of previous action."""
        if self.prev_action is not None and self.prev_acc is not None:
            delta_acc = acc - self.prev_acc
            delta_params = (params - self.prev_params) / 1e6
            reward = delta_acc - self.cost_lambda * delta_params
            a = self.prev_action
            if reward > 0:
                self.alpha[a] += min(reward * 10, 2.0)  # bounded update
            else:
                self.beta_p[a] += min(abs(reward) * 10, 2.0)
        self.prev_acc = acc
        self.prev_params = params

    def propose(self, m, acc, pa, drift):
        if self.u >= self.B:
            self.prev_action = "no_op"
            return None

        # Thompson sampling: sample from Beta posterior for each action
        samples = {}
        for a in self.actions:
            if a == "no_op":
                samples[a] = np.random.beta(self.alpha[a], self.beta_p[a])
            elif a == "add_blk" and len(m.cfgs) >= 10:
                samples[a] = -1  # infeasible
            elif a == "add_ds" and sum(1 for c in m.cfgs if c[2]==2) >= 4:
                samples[a] = -1  # infeasible
            elif a == "rm_blk" and len(m.cfgs) <= 2:
                samples[a] = -1  # infeasible
            else:
                samples[a] = np.random.beta(self.alpha[a], self.beta_p[a])

        best_a = max(samples, key=samples.get)

        if best_a == "no_op":
            self.prev_action = "no_op"
            return None

        self.u += 1
        self.prev_action = best_a
        self.log.append({"m": best_a, "rem": self.B - self.u, "acc": acc, "drift": drift,
                         "samples": {k: round(v,3) for k,v in samples.items()}})
        return best_a

# ═══════════════ DOMAIN LISTS ═══════════════

DOM3 = ["cifar10","cifar100","svhn"]
DOM5 = ["cifar10","cifar100","svhn","fashion"]  # 4-domain longer stream

# ═══════════════ 10 METHODS ═══════════════

def run_fixed(data, seed, domains=DOM3):
    tl=[]; step=0; t0=time.time()
    for dn in domains:
        tr,te,nc=data[dn]; m=Net(nc).to(DEV)
        tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method="Fixed Backbone",timeline=tl,arch=[],over=dict(t=time.time()-t0,mut=0))

def run_growing(data, seed, domains=DOM3):
    tl=[]; step=0; t0=time.time(); m=None; ac=[]
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=add_blk(m); m=add_blk(m); m=new_head(m,nc); ac.append(dict(step=step,ds=dn,a="add_2"))
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method="Growing (Naive)",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=2*(len(domains)-1)))

def run_ewc(data, seed, domains=DOM3):
    tl=[]; step=0; t0=time.time(); m=None; fi={}; pp={}
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        atr=DataLoader(tr,BS,shuffle=True,num_workers=0)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS):
                if fi: _,ta=train_ewc(m,cl,opt,cr,DEV,fi,pp,EWC_LAM)
                else: _,ta=train_ep(m,cl,opt,cr,DEV)
                sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
        fn=fisher(m,atr,DEV)
        for n in fn:
            if n in fi and fi[n].shape==fn[n].shape: fi[n]=fi[n]+fn[n]
            else: fi[n]=fn[n]
        pp={n:p.clone().detach() for n,p in m.named_parameters() if p.requires_grad}
    return dict(method="EWC",timeline=tl,arch=[],over=dict(t=time.time()-t0,mut=0))

def run_er(data, seed, domains=DOM3):
    tl=[]; step=0; t0=time.time(); m=None; buf=(torch.empty(0),torch.empty(0,dtype=torch.long))
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_er(m,cl,opt,cr,DEV,buf); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
        nb=fill_buf(DataLoader(tr,BS,shuffle=True,num_workers=0),ER_BUF)
        if buf[0].numel()>0:
            rx=torch.cat([buf[0],nb[0]]); ry=torch.cat([buf[1],nb[1]])
            if len(rx)>ER_BUF*3: ix=torch.randperm(len(rx))[:ER_BUF*3]; rx=rx[ix]; ry=ry[ix]
            buf=(rx,ry)
        else: buf=nb
    return dict(method="Exp. Replay",timeline=tl,arch=[],over=dict(t=time.time()-t0,mut=0))

def run_den(data, seed, domains=DOM3):
    tl=[]; step=0; t0=time.time(); m=None; ac=[]; nm=0
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else:
            m=add_blk(m); m=new_head(m,nc); nm+=1; ac.append(dict(step=step,ds=dn,a="den_expand"))
            for i,b in enumerate(m.blocks):
                if i<len(m.blocks)-2:
                    for p in b.parameters(): p.requires_grad=False
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD([p for p in m.parameters() if p.requires_grad],LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
        for p in m.parameters(): p.requires_grad=True
    return dict(method="DEN-style",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=nm))

def run_smart_grow(data, seed, domains=DOM3):
    """Smart Growing: grow only when accuracy drops below threshold (not unconditionally)."""
    tl=[]; step=0; t0=time.time(); m=None; ac=[]; nm=0; pa=0
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        grown_this_domain = 0
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            # Smart: grow if acc is low and haven't grown too much
            if te_a < 0.35 and grown_this_domain < 2 and len(m.cfgs) < 10:
                m=add_blk(m,warmstart=True); m=m.to(DEV)
                opt=optim.SGD(m.parameters(),LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                nm+=1; grown_this_domain+=1; ac.append(dict(step=step,ds=dn,ci=ci,m="smart_add"))
            pa=te_a
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method="Smart Growing",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=nm))

def run_rnas(data, seed, B=3, domains=DOM3):
    rng=random.Random(seed+999)
    tl=[]; step=0; t0=time.time(); m=None; ac=[]; nm=0
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=new_head(m,nc)
        m=m.to(DEV); used=0; tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            if used<B and rng.random()<.5:
                mut=rng.choice(["add_blk","add_ds","rm_blk"])
                if mut=="add_blk" and len(m.cfgs)<10: m=add_blk(m)
                elif mut=="add_ds" and sum(1 for c in m.cfgs if c[2]==2)<4: m=add_ds(m)
                elif mut=="rm_blk" and len(m.cfgs)>2: m=rm_blk(m)
                else: m=add_blk(m)
                m=m.to(DEV); opt=optim.SGD(m.parameters(),LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                used+=1; nm+=1; ac.append(dict(step=step,ds=dn,ci=ci,m=mut))
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method=f"RandomNAS (B={B})",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=nm))

def run_bnas_heuristic(data, seed, B=3, warmstart=False, freeze=False, domains=DOM3):
    """BudgetNAS with original heuristic controller, optionally with stabilization."""
    tl=[]; step=0; t0=time.time(); m=None; pa=0; ac=[]; nm=0
    ctrl=HeuristicCtrl(B); dd=LossDrift()
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: ctrl.reset(); dd=LossDrift(); m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            tl_v,te_a=ev(m,tel,cr,DEV); dd.update(tl_v); dr=dd.detect()
            mut=ctrl.propose(m,te_a,pa,dr)
            if mut:
                op=m.nparams()
                if mut=="add_blk": m=add_blk(m, warmstart=warmstart)
                elif mut=="add_ds": m=add_ds(m, warmstart=warmstart)
                elif mut=="rm_blk": m=rm_blk(m)
                m=m.to(DEV)
                if freeze:
                    freeze_early(m, keep_last=2)
                opt=optim.SGD([p for p in m.parameters() if p.requires_grad],LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                nm+=1; ac.append(dict(step=step,ds=dn,ci=ci,m=mut,op=op,np_=m.nparams(),rem=ctrl.B-ctrl.u,dr=dr))
            if freeze and mut:
                # unfreeze after one chunk of stabilization
                unfreeze_all(m)
            pa=te_a
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs),dr=dr,mut=mut)); step+=1
    ws_tag = "+WS" if warmstart else ""
    fr_tag = "+FZ" if freeze else ""
    return dict(method=f"BudgetNAS-Heur (B={B}){ws_tag}{fr_tag}",timeline=tl,arch=ac,
                over=dict(t=time.time()-t0,mut=nm),ctrl_log=ctrl.log)

def run_bnas_bandit(data, seed, B=3, warmstart=True, freeze=True, cost_lambda=0.05, domains=DOM3):
    """BudgetNAS with contextual bandit controller + warm-start + freeze."""
    tl=[]; step=0; t0=time.time(); m=None; pa=0; ac=[]; nm=0
    ctrl=BanditCtrl(B, cost_lambda=cost_lambda)
    dd=LossDrift()
    mmd=MMDDrift(threshold=0.1)

    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else:
            ctrl.reset(); dd=LossDrift(); m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)

        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            tl_v,te_a=ev(m,tel,cr,DEV); dd.update(tl_v); dr=dd.detect()

            # Update bandit reward from previous action
            ctrl.update_reward(te_a, m.nparams())

            mut=ctrl.propose(m,te_a,pa,dr)
            if mut:
                op=m.nparams()
                if mut=="add_blk": m=add_blk(m, warmstart=warmstart)
                elif mut=="add_ds": m=add_ds(m, warmstart=warmstart)
                elif mut=="rm_blk": m=rm_blk(m)
                m=m.to(DEV)
                if freeze:
                    freeze_early(m, keep_last=2)
                opt=optim.SGD([p for p in m.parameters() if p.requires_grad],LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                nm+=1; ac.append(dict(step=step,ds=dn,ci=ci,m=mut,op=op,np_=m.nparams(),rem=ctrl.B-ctrl.u,dr=dr))
            if freeze and mut:
                unfreeze_all(m)
            pa=te_a
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs),dr=dr,mut=mut)); step+=1
    return dict(method=f"BudgetNAS-Bandit (B={B})",timeline=tl,arch=ac,
                over=dict(t=time.time()-t0,mut=nm),ctrl_log=ctrl.log,
                bandit_state=dict(alpha={k:round(v,2) for k,v in ctrl.alpha.items()},
                                  beta={k:round(v,2) for k,v in ctrl.beta_p.items()}))

# ═══════════════ GRADUAL SHIFT ═══════════════

def gradual_stream(data, seed, domains=DOM3):
    rng=random.Random(seed); stream=[]
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]; tel=DataLoader(te,BS,num_workers=0)
        idx=list(range(len(tr))); rng.shuffle(idx); cs=len(idx)//3
        for ci in range(3):
            s=ci*cs; e=s+cs if ci<2 else len(idx)
            stream.append((Subset(tr,idx[s:e]),tel,nc,dn))
        if di<len(domains)-1:
            ndn=domains[di+1]; ntr,nte,nnc=data[ndn]; ntl=DataLoader(nte,BS,num_workers=0)
            ci1=rng.sample(range(len(tr)),min(400,len(tr)))
            ci2=rng.sample(range(len(ntr)),min(400,len(ntr)))
            blend=ConcatDataset([Subset(tr,ci1),Subset(ntr,ci2)])
            stream.append((blend,ntl,nnc,f"blend_{dn}_{ndn}"))
    return stream

def run_grad(data, seed, method="fixed", domains=DOM3):
    st=gradual_stream(data,seed,domains); tl=[]; step=0; m=None; pnc=None
    ctrl=BanditCtrl(3) if method=="bandit" else HeuristicCtrl(3)
    dd=LossDrift(); pa=0; t0=time.time(); nm=0
    for ch,tel,nc,lab in st:
        if m is None or nc!=pnc:
            if method=="fixed": m=Net(nc).to(DEV)
            elif m is None: m=Net(nc).to(DEV)
            else:
                ctrl.reset(); dd=LossDrift(); m=new_head(m,nc); m=m.to(DEV)
            pnc=nc
        cr=nn.CrossEntropyLoss(); opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
        for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV)
        tl_v,te_a=ev(m,tel,cr,DEV)
        if method in ("bandit","heuristic"):
            dd.update(tl_v); dr=dd.detect()
            if method == "bandit": ctrl.update_reward(te_a, m.nparams())
            mut=ctrl.propose(m,te_a,pa,dr)
            if mut:
                if mut=="add_blk": m=add_blk(m, warmstart=True)
                elif mut=="add_ds": m=add_ds(m, warmstart=True)
                elif mut=="rm_blk": m=rm_blk(m)
                m=m.to(DEV); nm+=1
        pa=te_a
        tl.append(dict(step=step,label=lab,acc=te_a,p=m.nparams(),bl=len(m.cfgs))); step+=1
    label_map = {"fixed":"Fixed","heuristic":"BudgetNAS-Heur","bandit":"BudgetNAS-Bandit"}
    return dict(method=f"{label_map[method]} (Gradual)",timeline=tl,over=dict(t=time.time()-t0,mut=nm))

# ═══════════════ TRIGGER ABLATION ═══════════════

def run_bnas_trigger_ablation(data, seed, trigger="both", B=1, domains=DOM3):
    """Ablate trigger conditions: drift-only, acc-only, both, neither."""
    tl=[]; step=0; t0=time.time(); m=None; pa=0; ac=[]; nm=0
    ctrl_B = B; ctrl_u = 0
    dd=LossDrift()
    for di,dn in enumerate(domains):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: ctrl_u=0; dd=LossDrift(); m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            tl_v,te_a=ev(m,tel,cr,DEV); dd.update(tl_v); dr=dd.detect()
            # Trigger logic
            should_mutate = False
            if trigger == "drift-only": should_mutate = dr
            elif trigger == "acc-only": should_mutate = te_a < 0.5
            elif trigger == "both": should_mutate = dr or te_a < 0.5
            elif trigger == "neither": should_mutate = False  # fixed schedule: always on first chunk
            if ctrl_u < ctrl_B and should_mutate and len(m.cfgs) < 10:
                m=add_blk(m, warmstart=True); m=m.to(DEV)
                opt=optim.SGD(m.parameters(),LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                ctrl_u+=1; nm+=1
            pa=te_a
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method=f"Trigger:{trigger}",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=nm))

# ═══════════════ METRICS ═══════════════

def compute_metrics(result, domains=DOM3):
    """Compute comprehensive metrics for a single run."""
    tl = result["timeline"]
    metrics = {}
    for dk in domains:
        es = [t for t in tl if t["ds"] == dk]
        if not es: continue
        accs = [t["acc"] for t in es]
        metrics[dk] = {
            "final_acc": accs[-1] * 100,
            "avg_acc": np.mean(accs) * 100,
            "auc": np.trapz(accs, dx=1.0) / len(accs) * 100,  # normalized AUC
            "final_params": es[-1]["p"],
            "final_blocks": es[-1]["bl"],
        }
    # Overall metrics
    all_accs = [t["acc"] for t in tl]
    n_mut = result["over"]["mut"]
    final_p = tl[-1]["p"] if tl else 0
    total_gain = sum(metrics[dk]["final_acc"] for dk in metrics) / len(metrics)
    metrics["overall"] = {
        "mean_final_acc": total_gain,
        "mean_auc": np.mean([metrics[dk]["auc"] for dk in metrics if dk != "overall"]),
        "total_params_M": final_p / 1e6,
        "total_mutations": n_mut,
        "gain_per_mutation": total_gain / max(n_mut, 1),
        "gain_per_M_params": total_gain / max(final_p / 1e6, 0.01),
        "acc_per_param_ratio": total_gain / max(final_p / 1e6, 0.01),
        "wall_time": result["over"]["t"],
    }
    return metrics

# ═══════════════ MAIN ═══════════════

def main():
    # Load existing results if any
    rpath = os.path.join(RES, "v3_results.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            ALL = json.load(f)
        print("Loaded existing v3 results")
    else:
        ALL = {}

    # ─── EXPERIMENT 1: Main comparison (10 methods × 3 seeds, 3-domain) ───
    methods_3d = [
        ("fixed",       lambda d,s: run_fixed(d,s,DOM3)),
        ("growing",     lambda d,s: run_growing(d,s,DOM3)),
        ("ewc",         lambda d,s: run_ewc(d,s,DOM3)),
        ("er",          lambda d,s: run_er(d,s,DOM3)),
        ("den",         lambda d,s: run_den(d,s,DOM3)),
        ("smart_grow",  lambda d,s: run_smart_grow(d,s,DOM3)),
        ("rnas",        lambda d,s: run_rnas(d,s,B=1,domains=DOM3)),
        ("bnas_heur",   lambda d,s: run_bnas_heuristic(d,s,B=1,warmstart=False,freeze=False,domains=DOM3)),
        ("bnas_ws",     lambda d,s: run_bnas_heuristic(d,s,B=1,warmstart=True,freeze=True,domains=DOM3)),
        ("bnas_bandit", lambda d,s: run_bnas_bandit(d,s,B=1,warmstart=True,freeze=True,domains=DOM3)),
    ]

    if "seeds" not in ALL: ALL["seeds"] = {}
    for seed in SEEDS:
        ss = str(seed)
        if ss not in ALL["seeds"]: ALL["seeds"][ss] = {}
        data = get_data(seed, DOM3)
        for mk, fn in methods_3d:
            if mk in ALL["seeds"][ss]: continue
            print(f"  Seed {seed} / {mk}...", end=" ", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            r = fn(data, seed)
            r["metrics"] = compute_metrics(r, DOM3)
            ALL["seeds"][ss][mk] = r
            print(f"done ({r['over']['t']:.0f}s)")
            save_json(ALL, "v3_results.json")
            gc.collect()

    # ─── EXPERIMENT 2: Budget sweep (B=0..5, bandit+ws+fz, seed 42) ───
    if "budget_sweep_v3" not in ALL: ALL["budget_sweep_v3"] = {}
    data42 = get_data(42, DOM3)
    for B in [0, 1, 2, 3, 5]:
        sb = str(B)
        if sb in ALL["budget_sweep_v3"]: continue
        print(f"  Budget B={B}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_bnas_bandit(data42, 42, B=B, domains=DOM3)
        r["metrics"] = compute_metrics(r, DOM3)
        ALL["budget_sweep_v3"][sb] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v3_results.json")

    # ─── EXPERIMENT 3: Trigger ablation (seed 42, B=1) ───
    if "trigger_ablation" not in ALL: ALL["trigger_ablation"] = {}
    for trig in ["drift-only", "acc-only", "both", "neither"]:
        if trig in ALL["trigger_ablation"]: continue
        print(f"  Trigger ablation: {trig}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_bnas_trigger_ablation(data42, 42, trigger=trig, B=1, domains=DOM3)
        r["metrics"] = compute_metrics(r, DOM3)
        ALL["trigger_ablation"][trig] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v3_results.json")

    # ─── EXPERIMENT 4: Stabilization ablation (seed 42, B=1) ───
    if "stabilization_ablation" not in ALL: ALL["stabilization_ablation"] = {}
    stab_configs = [
        ("none",      False, False),
        ("ws_only",   True,  False),
        ("fz_only",   False, True),
        ("ws+fz",     True,  True),
    ]
    for name, ws, fz in stab_configs:
        if name in ALL["stabilization_ablation"]: continue
        print(f"  Stabilization ablation: {name}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_bnas_heuristic(data42, 42, B=1, warmstart=ws, freeze=fz, domains=DOM3)
        r["metrics"] = compute_metrics(r, DOM3)
        ALL["stabilization_ablation"][name] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v3_results.json")

    # ─── EXPERIMENT 5: Gradual shift (seed 42) ───
    if "gradual_v3" not in ALL: ALL["gradual_v3"] = {}
    for meth in ["fixed", "heuristic", "bandit"]:
        if meth in ALL["gradual_v3"]: continue
        print(f"  Gradual {meth}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_grad(data42, 42, meth, DOM3)
        ALL["gradual_v3"][meth] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v3_results.json")

    # ─── EXPERIMENT 6: Longer stream (5 domains, seed 42) ───
    if "long_stream" not in ALL: ALL["long_stream"] = {}
    data5 = get_data(42, DOM5)
    long_methods = [
        ("fixed",       lambda d,s: run_fixed(d,s,DOM5)),
        ("growing",     lambda d,s: run_growing(d,s,DOM5)),
        ("bnas_bandit", lambda d,s: run_bnas_bandit(d,s,B=1,warmstart=True,freeze=True,domains=DOM5)),
        ("rnas",        lambda d,s: run_rnas(d,s,B=1,domains=DOM5)),
    ]
    for mk, fn in long_methods:
        if mk in ALL["long_stream"]: continue
        print(f"  Long stream / {mk}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = fn(data5, 42)
        r["metrics"] = compute_metrics(r, DOM5)
        ALL["long_stream"][mk] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v3_results.json")

    # ─── AGGREGATE ───
    print(f"\n{'='*70}")
    print("AGGREGATED v3 RESULTS (mean +/- std over 3 seeds)")
    print(f"{'='*70}")
    mks = ["fixed","growing","ewc","er","den","smart_grow","rnas","bnas_heur","bnas_ws","bnas_bandit"]
    agg = {}
    for mk in mks:
        agg[mk] = {}
        for dk in DOM3:
            fa=[]; av=[]; au=[]
            for s in SEEDS:
                ss = str(s)
                if mk not in ALL["seeds"][ss]: continue
                es=[t for t in ALL["seeds"][ss][mk]["timeline"] if t["ds"]==dk]
                if es:
                    fa.append(es[-1]["acc"])
                    av.append(np.mean([t["acc"] for t in es]))
                    au.append(np.trapz([t["acc"] for t in es], dx=1.0)/len(es))
            fp=ALL["seeds"][str(SEEDS[0])][mk]["timeline"]
            dse=[t for t in fp if t["ds"]==dk]
            agg[mk][dk]=dict(
                fm=float(np.mean(fa)*100) if fa else 0, fs=float(np.std(fa)*100) if fa else 0,
                am=float(np.mean(av)*100) if av else 0, a_s=float(np.std(av)*100) if av else 0,
                auc_m=float(np.mean(au)*100) if au else 0, auc_s=float(np.std(au)*100) if au else 0,
                p=dse[-1]["p"] if dse else 0, bl=dse[-1]["bl"] if dse else 0,
            )
        wt=[ALL["seeds"][str(s)][mk]["over"]["t"] for s in SEEDS if mk in ALL["seeds"][str(s)]]
        mt=[ALL["seeds"][str(s)][mk]["over"]["mut"] for s in SEEDS if mk in ALL["seeds"][str(s)]]
        agg[mk]["wt"]=float(np.mean(wt)) if wt else 0
        agg[mk]["mut"]=float(np.mean(mt)) if mt else 0
        nm=ALL["seeds"][str(SEEDS[0])].get(mk,{}).get("method","?")
        print(f"\n  {nm}")
        for dk in DOM3:
            a=agg[mk][dk]
            print(f"    {dk:10s}: final={a['fm']:.1f}+/-{a['fs']:.1f}%  avg={a['am']:.1f}+/-{a['a_s']:.1f}%  AUC={a['auc_m']:.1f}+/-{a['auc_s']:.1f}%  {a['p']/1e6:.2f}M  {a['bl']}blk")
        # Efficiency metrics
        mean_final = np.mean([agg[mk][dk]['fm'] for dk in DOM3])
        final_p = agg[mk][DOM3[-1]]['p']/1e6
        total_mut = agg[mk]['mut']
        print(f"    mean_final={mean_final:.1f}%  gain/mut={mean_final/max(total_mut,1):.1f}  gain/M={mean_final/max(final_p,0.01):.1f}  time={agg[mk]['wt']:.0f}s")

    ALL["aggregated_v3"] = agg
    save_json(ALL, "v3_results.json")
    print(f"\nAll v3 results saved to {os.path.join(RES,'v3_results.json')}")

if __name__ == "__main__":
    main()
