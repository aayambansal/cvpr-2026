#!/usr/bin/env python3
"""
BudgetNAS v2 — FAST incremental runner
=======================================
Saves after every method+seed so we never lose progress.
7 methods × 3 seeds + budget sweep + gradual shift + ablation.
"""
import os, sys, json, time, copy, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision, torchvision.transforms as T

# ── device ──
if torch.backends.mps.is_available(): DEV = torch.device("mps")
elif torch.cuda.is_available(): DEV = torch.device("cuda")
else: DEV = torch.device("cpu")
print(f"Device: {DEV}")

# ── config (tuned for MPS speed) ──
BS      = 256
EPOCHS  = 3          # per chunk
CHUNKS  = 4          # per dataset
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

# ── data ──
def tfm(name):
    m = ((.4914,.4822,.4465),(.2023,.1994,.2010)) if name!="svhn" else ((.4377,.4438,.4728),(.198,.201,.197))
    tr = T.Compose([T.RandomCrop(32,4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*m)])
    te = T.Compose([T.ToTensor(), T.Normalize(*m)])
    return tr, te

def sub(ds, n, seed):
    if len(ds) <= n: return ds
    return Subset(ds, random.Random(seed).sample(range(len(ds)), n))

_cache = {}
def get_data(seed):
    if seed in _cache: return _cache[seed]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    d = {}
    for name, cls, kw in [("cifar10", torchvision.datasets.CIFAR10, {}),
                           ("cifar100", torchvision.datasets.CIFAR100, {}),
                           ("svhn", torchvision.datasets.SVHN, {"split":"train"})]:
        tr_t, te_t = tfm(name)
        kw_te = {"split":"test"} if name=="svhn" else {}
        train = sub(cls(root, download=True, transform=tr_t, **({} if name!="svhn" else {"split":"train"})), N_TRAIN, seed)
        test  = sub(cls(root, download=True, transform=te_t, **({} if name!="svhn" else {"split":"test"})), N_TEST, seed)
        nc = 100 if name=="cifar100" else 10
        d[name] = (train, test, nc)
    _cache[seed] = d
    return d

def chunks(ds, seed):
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    cs = len(idx)//CHUNKS
    return [Subset(ds, idx[i*cs:(i+1)*cs if i<CHUNKS-1 else len(idx)]) for i in range(CHUNKS)]

# ── model ──
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
    def nparams(self): return sum(p.numel() for p in self.parameters())

def add_blk(m):
    c=list(m.cfgs); lc=c[-1][1]; c.append((lc,lc,1)); return _reb(m,c)
def add_ds(m):
    c=list(m.cfgs); lc=c[-1][1]; c.append((lc,min(lc*2,256),2)); return _reb(m,c)
def rm_blk(m):
    if len(m.cfgs)<=2: return m
    c=list(m.cfgs); c.pop(-2)
    prev=32
    fc=[]
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

# ── training ──
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

# ── controller ──
class Ctrl:
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

class Drift:
    def __init__(self,w=2,t=.25): self.w=w; self.t=t; self.h=[]
    def up(self,l): self.h.append(l)
    def det(self):
        if len(self.h)<self.w+1: return False
        r=np.mean(self.h[-self.w:]); o=np.mean(self.h[-2*self.w:-self.w]) if len(self.h)>=2*self.w else self.h[0]
        return (r-o)/(o+1e-8)>self.t

DOM = ["cifar10","cifar100","svhn"]

# ═══════════════ 7 METHODS ═══════════════

def run_fixed(data, seed):
    tl=[]; step=0; t0=time.time()
    for dn in DOM:
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

def run_growing(data, seed):
    tl=[]; step=0; t0=time.time(); m=None; ac=[]
    for di,dn in enumerate(DOM):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: m=add_blk(add_blk(m)); m=new_head(m,nc); ac.append(dict(step=step,ds=dn,a="add_2"))
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            _,te_a=ev(m,tel,cr,DEV)
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method="Growing (Naive)",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=4))

def run_ewc(data, seed):
    tl=[]; step=0; t0=time.time(); m=None; fi={}; pp={}
    for di,dn in enumerate(DOM):
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

def run_er(data, seed):
    tl=[]; step=0; t0=time.time(); m=None; buf=(torch.empty(0),torch.empty(0,dtype=torch.long))
    for di,dn in enumerate(DOM):
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

def run_den(data, seed):
    tl=[]; step=0; t0=time.time(); m=None; ac=[]; nm=0
    for di,dn in enumerate(DOM):
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

def run_rnas(data, seed, B=3):
    rng=random.Random(seed+999)
    tl=[]; step=0; t0=time.time(); m=None; ac=[]; nm=0
    for di,dn in enumerate(DOM):
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

def run_bnas(data, seed, B=3):
    tl=[]; step=0; t0=time.time(); m=None; pa=0; ac=[]; nm=0
    ctrl=Ctrl(B); dd=Drift()
    for di,dn in enumerate(DOM):
        tr,te,nc=data[dn]
        if m is None: m=Net(nc)
        else: ctrl.reset(); dd=Drift(); m=new_head(m,nc)
        m=m.to(DEV); tel=DataLoader(te,BS,num_workers=0); cr=nn.CrossEntropyLoss()
        opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,CHUNKS*EPOCHS)
        for ci,ch in enumerate(chunks(tr,seed)):
            cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
            for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV); sch.step()
            tl_v,te_a=ev(m,tel,cr,DEV); dd.up(tl_v); dr=dd.det()
            mut=ctrl.propose(m,te_a,pa,dr)
            if mut:
                op=m.nparams()
                if mut=="add_blk": m=add_blk(m)
                elif mut=="add_ds": m=add_ds(m)
                elif mut=="rm_blk": m=rm_blk(m)
                m=m.to(DEV); opt=optim.SGD(m.parameters(),LR*.5,momentum=.9,weight_decay=WD)
                sch=optim.lr_scheduler.CosineAnnealingLR(opt,max(1,(CHUNKS-ci)*EPOCHS))
                nm+=1; ac.append(dict(step=step,ds=dn,ci=ci,m=mut,op=op,np=m.nparams(),rem=ctrl.B-ctrl.u,dr=dr))
            pa=te_a
            tl.append(dict(step=step,ds=dn,chunk=ci,acc=te_a,ta=ta,p=m.nparams(),bl=len(m.cfgs),dr=dr,mut=mut)); step+=1
    return dict(method=f"BudgetNAS (B={B})",timeline=tl,arch=ac,over=dict(t=time.time()-t0,mut=nm),ctrl_log=ctrl.log)

# ═══════════════ GRADUAL SHIFT ═══════════════
def gradual_stream(data, seed):
    """3 chunks per domain + 1 blended chunk between domains."""
    rng=random.Random(seed); stream=[]
    for di,dn in enumerate(DOM):
        tr,te,nc=data[dn]; tel=DataLoader(te,BS,num_workers=0)
        idx=list(range(len(tr))); rng.shuffle(idx); cs=len(idx)//3
        for ci in range(3):
            s=ci*cs; e=s+cs if ci<2 else len(idx)
            stream.append((Subset(tr,idx[s:e]),tel,nc,dn))
        if di<len(DOM)-1:
            ndn=DOM[di+1]; ntr,nte,nnc=data[ndn]; ntl=DataLoader(nte,BS,num_workers=0)
            ci1=rng.sample(range(len(tr)),min(400,len(tr)))
            ci2=rng.sample(range(len(ntr)),min(400,len(ntr)))
            blend=ConcatDataset([Subset(tr,ci1),Subset(ntr,ci2)])
            stream.append((blend,ntl,nnc,f"blend_{dn}_{ndn}"))
    return stream

def run_grad(data, seed, method="fixed"):
    st=gradual_stream(data,seed); tl=[]; step=0; m=None; pnc=None
    ctrl=Ctrl(3); dd=Drift(); pa=0; t0=time.time()
    for ch,tel,nc,lab in st:
        if m is None or nc!=pnc:
            if method=="fixed": m=Net(nc).to(DEV)
            elif m is None: m=Net(nc).to(DEV)
            else:
                ctrl.reset(); dd=Drift(); m=new_head(m,nc); m=m.to(DEV)
            pnc=nc
        cr=nn.CrossEntropyLoss(); opt=optim.SGD(m.parameters(),LR,momentum=.9,weight_decay=WD)
        cl=DataLoader(ch,BS,shuffle=True,num_workers=0); ta=0
        for _ in range(EPOCHS): _,ta=train_ep(m,cl,opt,cr,DEV)
        tl_v,te_a=ev(m,tel,cr,DEV)
        if method=="bnas":
            dd.up(tl_v); dr=dd.det()
            mut=ctrl.propose(m,te_a,pa,dr)
            if mut:
                if mut=="add_blk": m=add_blk(m)
                elif mut=="add_ds": m=add_ds(m)
                elif mut=="rm_blk": m=rm_blk(m)
                m=m.to(DEV)
        pa=te_a
        tl.append(dict(step=step,label=lab,acc=te_a,p=m.nparams(),bl=len(m.cfgs))); step+=1
    return dict(method=f"{'Fixed' if method=='fixed' else 'BudgetNAS'} (Gradual)",timeline=tl,over=dict(t=time.time()-t0))

# ═══════════════ MAIN ═══════════════
def main():
    ALL = {}
    methods = [
        ("fixed", run_fixed),
        ("growing", run_growing),
        ("ewc", run_ewc),
        ("er", run_er),
        ("den", run_den),
        ("rnas", lambda d,s: run_rnas(d,s,3)),
        ("bnas", lambda d,s: run_bnas(d,s,3)),
    ]

    # ── main experiment: 7 methods × 3 seeds ──
    ALL["seeds"] = {}
    for seed in SEEDS:
        print(f"\n{'='*50} SEED {seed} {'='*50}")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        data = get_data(seed)
        ALL["seeds"][seed] = {}
        for mk, fn in methods:
            print(f"  {mk}...", end=" ", flush=True)
            r = fn(data, seed)
            ALL["seeds"][seed][mk] = r
            print(f"done ({r['over']['t']:.0f}s)")
            save_json(ALL, "v2_results.json")  # incremental save
            gc.collect()

    # ── budget sweep (seed 42 only) ──
    print(f"\n{'='*50} BUDGET SWEEP {'='*50}")
    data42 = get_data(42)
    ALL["budget_sweep"] = {}
    for B in [0,1,2,3,5]:
        print(f"  B={B}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_bnas(data42, 42, B=B)
        ALL["budget_sweep"][B] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v2_results.json")

    # ── gradual shift (seed 42) ──
    print(f"\n{'='*50} GRADUAL SHIFT {'='*50}")
    ALL["gradual"] = {}
    for meth in ["fixed","bnas"]:
        print(f"  {meth}...", end=" ", flush=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        r = run_grad(data42, 42, meth)
        ALL["gradual"][meth] = r
        print(f"done ({r['over']['t']:.0f}s)")
        save_json(ALL, "v2_results.json")

    # ── AGGREGATE ──
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean ± std over 3 seeds)")
    print(f"{'='*70}")
    mks = ["fixed","growing","ewc","er","den","rnas","bnas"]
    agg = {}
    for mk in mks:
        agg[mk] = {}
        for dk in DOM:
            fa=[]; av=[]
            for s in SEEDS:
                es=[t for t in ALL["seeds"][s][mk]["timeline"] if t["ds"]==dk]
                if es: fa.append(es[-1]["acc"]); av.append(np.mean([t["acc"] for t in es]))
            fp=ALL["seeds"][SEEDS[0]][mk]["timeline"]
            dse=[t for t in fp if t["ds"]==dk]
            agg[mk][dk]=dict(fm=np.mean(fa)*100, fs=np.std(fa)*100, am=np.mean(av)*100,
                             a_s=np.std(av)*100, p=dse[-1]["p"] if dse else 0, bl=dse[-1]["bl"] if dse else 0)
        wt=[ALL["seeds"][s][mk]["over"]["t"] for s in SEEDS]
        mt=[ALL["seeds"][s][mk]["over"]["mut"] for s in SEEDS]
        agg[mk]["wt"]=np.mean(wt); agg[mk]["mut"]=np.mean(mt)
        nm=ALL["seeds"][SEEDS[0]][mk]["method"]
        print(f"\n  {nm}")
        for dk in DOM:
            a=agg[mk][dk]
            print(f"    {dk:10s}: {a['fm']:.1f}±{a['fs']:.1f}%  avg={a['am']:.1f}±{a['a_s']:.1f}%  {a['p']/1e6:.2f}M  {a['bl']}blk")
        print(f"    time={agg[mk]['wt']:.0f}s  muts={agg[mk]['mut']:.0f}")

    ALL["aggregated"] = agg

    # Budget sweep summary
    print(f"\n{'='*50} BUDGET SWEEP {'='*50}")
    for B in [0,1,2,3,5]:
        r=ALL["budget_sweep"][B]["timeline"]
        line = f"  B={B}: "
        for dk in DOM:
            es=[t for t in r if t["ds"]==dk]
            if es: line+=f"{dk}={es[-1]['acc']*100:.1f}% "
        print(line)

    save_json(ALL, "v2_results.json")
    print(f"\nAll results saved to {os.path.join(RES,'v2_results.json')}")

if __name__=="__main__":
    main()
