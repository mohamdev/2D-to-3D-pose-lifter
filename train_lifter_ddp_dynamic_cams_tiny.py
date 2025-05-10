#!/usr/bin/env python3
"""
train_dynamic_projection.py

Training with on-the-fly 2D projection, supports both single-GPU/CPU and DDP:
 - Reads only 3D poses
 - Samples random camera intrinsics/extrinsics per window
 - Projects 3D -> 2D online
 - Matches Transformer architecture from the old pipeline
"""
import os
import argparse
import glob
import bisect
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Try importing distributed modules
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler
    has_dist = True
except ImportError:
    has_dist = False

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1920, 1080
F_MIN, F_MAX = 800, 1400  # focal-length range (px)
D_MAX        = 6.0        # max camera distance (m)
MARGIN       = 1.1        # safety margin on fov
SEG_NORM     = 2.0        # segment-length normalization

J = 12      # joints
S = 6       # segments
D_IN = 11   # 2D + 3 intrinsics + 6 segments
D_MODEL = 128
N_HEAD = 4
N_LAYERS = 4
D_FF = 256

SKELETON_EDGES = [
    (0,2),(2,4),(1,3),(3,5),
    (6,8),(8,10),(7,9),(9,11),
    (0,1),(6,7),(0,6),(1,7),
]

# ─── UTILS FOR CAMERA SAMPLING ──────────────────────────────────────────────────
def sample_direction():
    v = np.random.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-8)

def look_at_R(cam_pos, target=np.zeros(3), up=np.array([0,1,0])):
    z = target - cam_pos
    z /= np.linalg.norm(z) + 1e-8
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)

# ─── DATASET ────────────────────────────────────────────────────────────────────
class DynamicPoseDataset(Dataset):
    def __init__(self, npz_dir, window, seed=0,
                 dist_min=1.5, dist_max=2.5):
        random.seed(seed)
        np.random.seed(seed)
        self.window = window
        self.W, self.H = IMG_W, IMG_H
        self.f_min, self.f_max = F_MIN, F_MAX
        self.d_max = D_MAX
        self.margin = MARGIN
        self.dist_min = dist_min
        self.dist_max = dist_max

        self.meta, self.cum = [], []
        total = 0
        for path in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
            data = np.load(path)
            joints_3d = data['joints_3d']
            F, J, _ = joints_3d.shape
            n_w = max(F - window + 1, 0)
            if n_w > 0:
                self.meta.append({'j3d': joints_3d})
                self.cum.append(total)
                total += n_w
        if total == 0:
            raise RuntimeError(f'No clips >= {window} frames in {npz_dir}')
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cum, idx) - 1
        local = idx - self.cum[file_idx]
        f0, f1 = local, local + self.window
        joints = self.meta[file_idx]['j3d']

        J3 = joints[f0:f1]  # (T,J,3)
        j0 = joints[f0]
        pairs = [(2,4),(3,5),(0,1),(0,6),(6,8),(7,9)]
        segs = np.array([np.linalg.norm(j0[i]-j0[j]) for i,j in pairs], dtype=np.float32)
        segs /= SEG_NORM

        # intrinsics
        f = np.random.uniform(self.f_min, self.f_max)
        cx = self.W/2 + np.random.uniform(-0.03,0.03)*self.W
        cy = self.H/2 + np.random.uniform(-0.03,0.03)*self.H
        K = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)
        k_norm = np.array([f/2000.0, (cx-self.W/2)/self.W, (cy-self.H/2)/self.H], np.float32)

        # extrinsic
        coords = joints.reshape(-1,3)
        R_body = np.linalg.norm(coords,axis=1).max()
        min_z = coords[:,2].min()
        min_xy = R_body * (self.dist_min/2)
        while True:
            v = sample_direction()
            factor = random.uniform(self.dist_min, self.dist_max)
            cam_pos = v * (R_body * factor)
            if cam_pos[2] < min_z or np.linalg.norm(cam_pos[:2]) < min_xy:
                continue
            R_w2c = look_at_R(cam_pos)
            t_w2c = -R_w2c.dot(cam_pos)
            break

        X = J3.reshape(-1,3).T
        Xc = R_w2c @ X + t_w2c[:,None]
        uvw = K @ Xc
        uv = (uvw[:2]/(uvw[2:]+1e-8)).T.reshape(self.window, J, 2)

        return {
            'x2d': torch.from_numpy(uv.astype(np.float32)/self.W),
            'y3d': torch.from_numpy(J3.astype(np.float32)),
            'k':   torch.from_numpy(k_norm),
            'seg': torch.from_numpy(segs),
            'K_pix': torch.from_numpy(K),
            'R_pix': torch.from_numpy(R_w2c.astype(np.float32)),
            't_pix': torch.from_numpy(t_w2c.astype(np.float32)),
        }

# ─── MODEL & LOSS ───────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x): return x + self.pe[:,:x.size(1)].to(x.device)

class TransformerLifter(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Linear(D_IN, D_MODEL)
        self.k_token     = nn.Parameter(torch.zeros(1,1,D_MODEL))
        self.seg_token   = nn.Parameter(torch.zeros(1,1,D_MODEL))
        self.k_embed     = nn.Linear(3, D_MODEL)
        self.seg_embed   = nn.Linear(S, D_MODEL)
        self.pos_enc     = PositionalEncoding(D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(d_model=D_MODEL,
                                               nhead=N_HEAD,
                                               dim_feedforward=D_FF,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.regressor   = nn.Linear(D_MODEL, 3)

    def forward(self, x2d, k, seg):
        B,T,J,_ = x2d.shape
        k_exp = k[:,None,None,:].expand(B,T,J,3)
        seg_exp = seg[:,None,None,:].expand(B,T,J,S)
        tokens = torch.cat([x2d, k_exp, seg_exp], dim=-1)
        tokens = tokens.reshape(B, T*J, D_IN)
        x = self.token_embed(tokens)
        k_proj    = self.k_embed(k)
        seg_proj  = self.seg_embed(seg)
        k_tok     = self.k_token.expand(B,-1,-1) + k_proj.unsqueeze(1)
        seg_tok   = self.seg_token.expand(B,-1,-1) + seg_proj.unsqueeze(1)
        x = torch.cat([k_tok, seg_tok, x], dim=1)
        x = self.pos_enc(x)
        out = self.transformer(x)
        out = out[:,2:,:].reshape(B,T,J,D_MODEL)
        return self.regressor(out)

def mpjpe(pred, target): return ((pred-target)**2).sum(-1).sqrt().mean()
def bone_lengths(x, pairs):
    return torch.stack([(x[:,:,i]-x[:,:,j]).norm(-1) for i,j in pairs], dim=-1)

def full_reprojection_loss(pred3d, x2d, K, R, t):
    B,T,J,_ = pred3d.shape
    pts = pred3d[:,0].reshape(B*J,3)
    R_b = R.unsqueeze(1).expand(B,J,3,3).reshape(-1,3,3)
    t_b = t.unsqueeze(1).expand(B,J,3).reshape(-1,3)
    cam = (R_b @ pts.unsqueeze(-1)).squeeze(-1) + t_b
    K_b = K.unsqueeze(1).expand(B,J,3,3).reshape(-1,3,3)
    uvw = (K_b @ cam.unsqueeze(-1)).squeeze(-1)
    uv = uvw[:,:2] / (uvw[:,2:]+1e-8)
    uv = uv.reshape(B,J,2)
    x_px = x2d[:,0] * IMG_W
    return (uv - x_px).abs().mean()

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch',  type=int, default=64)
    parser.add_argument('--window',type=int, default=13)
    parser.add_argument('--checkpoint',default=None)
    parser.add_argument('--amp',action='store_true')
    args = parser.parse_args()

    # init DDP or single
    if has_dist and 'RANK' in os.environ:
        dist.init_process_group('nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        rank = dist.get_rank(); world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank=0; rank=0; world_size=1

    # dataset
    full_ds = DynamicPoseDataset(args.data, window=args.window, seed=0)
    N = len(full_ds); N_val = int(0.05*N); N_train=N-N_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(full_ds, [N_train,N_val], generator=g)

    # loaders
    if world_size>1:
        train_sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler  =DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler=None; val_sampler=None
    nw=max(1, min(4, os.cpu_count()//2))
    train_loader=DataLoader(train_ds, batch_size=args.batch, sampler=train_sampler,
                             shuffle=(train_sampler is None), num_workers=nw, pin_memory=True)
    val_loader  =DataLoader(val_ds,   batch_size=args.batch, sampler=val_sampler,
                             shuffle=False, num_workers=nw, pin_memory=True)

    # model
    model = TransformerLifter().to(device)
    if world_size>1:
        model=DDP(model, device_ids=[local_rank])
    opt=optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler=torch.cuda.amp.GradScaler() if args.amp else None

    # resume
    start_epoch=1; best_val=float('inf')
    if args.checkpoint:
        ckpt=torch.load(args.checkpoint,map_location='cpu')
        sd=ckpt.get('model_state_dict', ckpt)
        tgt = model.module if hasattr(model, 'module') else model
        new_sd={k.replace('module.',''):v for k,v in sd.items()}
        tgt.load_state_dict(new_sd)
        if 'optimizer_state_dict' in ckpt: opt.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt: sched.load_state_dict(ckpt['scheduler_state_dict'])
        if args.amp and 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch=ckpt.get('epoch',0)+1
    if world_size>1:
        dist.barrier()

    # train/val loops
    for epoch in range(start_epoch, args.epochs+1):
        if world_size>1: train_sampler.set_epoch(epoch)
        model.train(); train_losses=[]
        pbar=tqdm(train_loader,desc=f"Epoch {epoch} [Train]",disable=(rank!=0),ncols=120)
        for batch in pbar:
            x2d = batch['x2d'].to(device)
            y3d = batch['y3d'].to(device)
            k   = batch['k'].to(device)
            seg = batch['seg'].to(device)
            Kpix= batch['K_pix'].to(device)
            Rpix= batch['R_pix'].to(device)
            tpix= batch['t_pix'].to(device)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x2d, k, seg)
                lm = mpjpe(pred, y3d)
                bp = bone_lengths(pred, SKELETON_EDGES)
                bt = bone_lengths(y3d, SKELETON_EDGES)
                lb = (bp-bt).abs().mean()
                lrp= full_reprojection_loss(pred, x2d, Kpix, Rpix, tpix)
                lr = lrp / IMG_W
                loss = lm + lb + 1e-2*lr
                # print("lb loss:", lb)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            train_losses.append((lm.item(), lb.item(), lr.item()))
            if rank==0:
                pbar.set_postfix({'MPJPE':f"{lm.item():.4f}",'Bone':f"{lb.item()}",'Reproj(px)':f"{lrp.item():.4f}"})
        sched.step()
        tm,tb,tr=np.mean(train_losses,axis=0)
        if rank==0:
            print(f"[Epoch {epoch}] Train MPJPE={tm:.4f}, Bone={tb:.4f}, Reproj={tr:.2f}")

        model.eval(); val_m=[]
        with torch.no_grad():
            pbar=tqdm(val_loader,desc=f"Epoch {epoch} [Val]",disable=(rank!=0),ncols=120)
            for batch in pbar:
                x2d = batch['x2d'].to(device)
                y3d = batch['y3d'].to(device)
                k   = batch['k'].to(device)
                seg = batch['seg'].to(device)
                Kpix= batch['K_pix'].to(device)
                Rpix= batch['R_pix'].to(device)
                tpix= batch['t_pix'].to(device)
                pred=model(x2d,k,seg)
                lm=mpjpe(pred,y3d)
                lb=(bone_lengths(pred,SKELETON_EDGES)-bone_lengths(y3d,SKELETON_EDGES)).abs().mean()
                lrp=full_reprojection_loss(pred,x2d,Kpix,Rpix,tpix)
                val_m.append((lm.item(),lb.item(),(lrp/IMG_W).item()))
        vm,vb,vr=np.mean(val_m,axis=0)
        if rank==0:
            print(f"[Epoch {epoch}] Val   MPJPE={vm:.4f}, Bone={vb:.4f}, Reproj={vr:.2f}")
            if vm<best_val:
                best_val=vm
                ckpt={'epoch':epoch,
                      'model_state_dict':model.module.state_dict() if hasattr(model,'module') else model.state_dict(),
                      'optimizer_state_dict':opt.state_dict(),
                      'scheduler_state_dict':sched.state_dict(),
                      'best_val':best_val}
                if scaler: ckpt['scaler_state_dict']=scaler.state_dict()
                torch.save(ckpt,f"best_lifter_{D_MODEL}_ddp_epoch{epoch}.pt")
                print(f"→ Saved best model (Val MPJPE={best_val:.4f})")
    if world_size>1: dist.destroy_process_group()

if __name__=='__main__':
    main()