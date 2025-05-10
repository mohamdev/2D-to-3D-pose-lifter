#!/usr/bin/env python3
import os
import argparse
import glob
import bisect
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
J        = 12      # joints
S        = 6       # segments
D_IN     = 11      # 2D + 3 intrinsics + 6 segment lengths
D_MODEL  = 128
N_HEAD   = 4
N_LAYERS = 4
D_FF     = 256
IMG_W    = 1920
IMG_H    = 1080

SKELETON_EDGES = [
    (0,2),(2,4),(1,3),(3,5),
    (6,8),(8,10),(7,9),(9,11),
    (0,1),(6,7),(0,6),(1,7),
]

# ─── DATASET ────────────────────────────────────────────────────────────────────
class AugmentedNPZDataset(Dataset):
    def __init__(self, npz_dir: str, window: int, return_extrinsics: bool = True):
        if dist.get_rank() == 0:
            print(f"[Dataset] Initializing from {npz_dir!r} with window={window}")
        self.window = int(window)
        self.W, self.H = IMG_W, IMG_H
        self.return_extrinsics = return_extrinsics

        self.meta = []
        self.cum_windows = []
        total = 0

        for path in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
            d = np.load(path, mmap_mode='r')
            P, F, _, _ = d['joints_2d'].shape
            n_f = max(F - self.window + 1, 0)
            n_w = P * n_f
            if n_w == 0:
                if dist.get_rank()==0:
                    print(f"[Dataset]  → skip {os.path.basename(path)} (too short)")
                continue
            if dist.get_rank()==0:
                print(f"[Dataset]  → file {os.path.basename(path)}: P={P}, F={F}, windows={n_w}")
            self.meta.append({'path': path, 'P':P, 'F':F, 'n_f':n_f, 'n_w':n_w})
            self.cum_windows.append(total)
            total += n_w

        self.total_windows = total
        if dist.get_rank()==0:
            print(f"[Dataset] Done init: {len(self.meta)} files, {self.total_windows} total windows.\n")
        if total == 0:
            raise RuntimeError(f'No windows found in “{npz_dir}”')

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx: int):
        file_idx = bisect.bisect_right(self.cum_windows, idx) - 1
        meta     = self.meta[file_idx]
        local    = idx - self.cum_windows[file_idx]
        p        = local // meta['n_f']
        f0       = local % meta['n_f']
        f1       = f0 + self.window

        if dist.get_rank()==0 and idx < 5:
            print(f"[Dataset]  idx={idx} → file#{file_idx}, cam={p}, f0={f0}")

        data = np.load(meta['path'], mmap_mode='r')
        x2d  = data['joints_2d'][p, f0:f1]     # (T,J,2)
        y3d  = data['joints_3d'][   f0:f1]     # (T,J,3)
        seg  = data['segments_lengths'][f0]    # (S,1)
        K    = data['K'][p]                    # (3,3)
        if self.return_extrinsics:
            R = data['R'][p]
            t = data['t'][p]

        # normalize intrinsics → k_vec
        f_norm  = K[0,0] / 2000.0
        cx_norm = (K[0,2] - self.W/2) / self.W
        cy_norm = (K[1,2] - self.H/2) / self.H
        k_vec   = np.asarray([f_norm, cx_norm, cy_norm], dtype=np.float32)

        sample = {
            'x2d'  : torch.from_numpy(x2d.astype(np.float32) / self.W),
            'y3d'  : torch.from_numpy(y3d.astype(np.float32)),
            'k'    : torch.from_numpy(k_vec),
            'seg'  : torch.from_numpy(seg.squeeze().astype(np.float32) / 2.0),
            'K_pix': torch.from_numpy(K.astype(np.float32)),
        }
        if self.return_extrinsics:
            sample['R_pix'] = torch.from_numpy(R.astype(np.float32))
            sample['t_pix'] = torch.from_numpy(t.astype(np.float32))

        return sample

# ─── MODEL ──────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerLifter(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Linear(D_IN,   D_MODEL)
        self.k_token     = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.seg_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.k_embed     = nn.Linear(3,      D_MODEL)
        self.seg_embed   = nn.Linear(6,      D_MODEL)
        self.pos_enc     = PositionalEncoding(D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD,
            dim_feedforward=D_FF, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.regressor   = nn.Linear(D_MODEL, 3)

    def forward(self, x2d, k, seg):
        B, T, J, _ = x2d.shape
        k_exp   = k[:,None,None,:].expand(B, T, J, 3)
        seg_exp = seg[:,None,None,:].expand(B, T, J, 6)
        tokens  = torch.cat([x2d, k_exp, seg_exp], dim=-1).reshape(B, T*J, D_IN)
        x       = self.token_embed(tokens)

        k_proj   = self.k_embed(k)
        seg_proj = self.seg_embed(seg)
        k_tok    = self.k_token.expand(B,-1,-1) + k_proj.unsqueeze(1)
        seg_tok  = self.seg_token.expand(B,-1,-1) + seg_proj.unsqueeze(1)
        x        = torch.cat([k_tok, seg_tok, x], dim=1)

        x = self.pos_enc(x)
        out = self.transformer(x)
        out = out[:,2:,:].reshape(B, T, J, D_MODEL)
        return self.regressor(out)

# ─── LOSS ───────────────────────────────────────────────────────────────────────
def mpjpe(pred, target):
    return ((pred - target)**2).sum(-1).sqrt().mean()

def bone_lengths(x, pairs):
    return torch.stack([(x[:,:,i]-x[:,:,j]).norm(dim=-1) for (i,j) in pairs], dim=-1)

def full_reprojection_loss(pred3d, x2d, K, R, t):
    B, T, J, _ = pred3d.shape
    pts = pred3d[:,0].reshape(B*J, 3)
    R_b = R.unsqueeze(1).expand(B, J, 3,3).reshape(-1,3,3)
    t_b = t.unsqueeze(1).expand(B, J, 3).reshape(-1,3)
    cam = (R_b @ pts.unsqueeze(-1)).squeeze(-1) + t_b
    K_b = K.unsqueeze(1).expand(B, J,3,3).reshape(-1,3,3)
    uvw = (K_b @ cam.unsqueeze(-1)).squeeze(-1)
    uv  = uvw[:,:2] / (uvw[:,2:]+1e-8)
    uv  = uv.reshape(B, J, 2)
    x2d_px = x2d[:,0] * IMG_W
    return (uv - x2d_px).abs().mean()

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # SLURM → DDP env
    if 'SLURM_PROCID' in os.environ:
        os.environ.setdefault('RANK',       os.environ['SLURM_PROCID'])
        os.environ.setdefault('WORLD_SIZE', os.environ['SLURM_NTASKS'])
        os.environ.setdefault('LOCAL_RANK', os.environ['SLURM_LOCALID'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, required=True,
                        help="Path to augmented_npz_grid/")
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch',      type=int, default=64)
    parser.add_argument('--window',     type=int, default=13,
                        help="Temporal window size")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a training checkpoint to resume from")
    parser.add_argument('--amp',        action='store_true')
    parser.add_argument('--save-every', type=int, default=0,
                        help="Save checkpoint every N train batches (0=off)")
    args = parser.parse_args()

    # Init DDP
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # Dataset & split
    full_ds = AugmentedNPZDataset(args.data, window=args.window, return_extrinsics=True)
    N       = len(full_ds)
    N_val   = int(0.1 * N)
    N_train = N - N_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(full_ds, [N_train, N_val], generator=g)

    # Samplers & loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
    n_workers = max(1, min(4, os.cpu_count()//2))
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=train_sampler,
                              num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              sampler=val_sampler,
                              num_workers=n_workers, pin_memory=True)

    # Model, optimizer, scheduler, scaler
    torch.backends.cudnn.benchmark = True
    model  = TransformerLifter().cuda(local_rank)
    model  = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt    = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Resume if requested (with step-level resume)
    start_epoch = 1
    best_val    = float('inf')
    already_seen = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        target = model.module if isinstance(model, DDP) else model
        # strip any "module." prefix
        new_sd = {k[len('module.'):] if k.startswith('module.') else k: v
                  for k,v in sd.items()}
        target.load_state_dict(new_sd)
        if 'optimizer_state_dict' in ckpt:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            sched.load_state_dict(ckpt['scheduler_state_dict'])
        if args.amp and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        # decide whether to re-do partial epoch or start next one
        if 'step' in ckpt:
            start_epoch   = ckpt['epoch']
            already_seen  = ckpt['step']
        else:
            start_epoch   = ckpt.get('epoch', 0) + 1

        best_val = ckpt.get('best_val', best_val)

    dist.barrier()

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=(rank!=0), ncols=120)
        for batch_idx, batch in enumerate(pbar, 1):
            # — skip already-seen batches on resume —
            if epoch == start_epoch and batch_idx <= already_seen:
                continue

            x2d, y3d, k, seg, K_pix, R_pix, t_pix = (
                batch['x2d'].cuda(local_rank),
                batch['y3d'].cuda(local_rank),
                batch['k'].cuda(local_rank),
                batch['seg'].cuda(local_rank),
                batch['K_pix'].cuda(local_rank),
                batch['R_pix'].cuda(local_rank),
                batch['t_pix'].cuda(local_rank),
            )

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred      = model(x2d, k, seg)
                loss_m    = mpjpe(pred, y3d)
                bones_p   = bone_lengths(pred, SKELETON_EDGES)
                bones_t   = bone_lengths(y3d,  SKELETON_EDGES)
                loss_b    = (bones_p - bones_t).abs().mean()
                loss_r_px = full_reprojection_loss(pred, x2d, K_pix, R_pix, t_pix)
                loss_r    = loss_r_px / IMG_W
                loss      = loss_m + 1.0*loss_b + 0.0*loss_r

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            train_losses.append((loss_m.item(), loss_b.item(), loss_r.item()))

            # periodic inside-epoch save
            if args.save_every > 0 and batch_idx % args.save_every == 0 and rank == 0:
                torch.save({
                    'epoch': epoch,
                    'step': batch_idx,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'best_val': best_val,
                }, f"ckpt_epoch_{D_MODEL}_{epoch:02d}_step{batch_idx:06d}.pt")

            if rank == 0:
                pbar.set_postfix({
                    'MPJPE':      f"{loss_m.item():.4f}",
                    'Bone':       f"{loss_b.item():.4f}",
                    'Reproj(px)': f"{loss_r_px.item():.4f}",
                    'Reproj(n)':  f"{loss_r.item():.4f}",
                })

        sched.step()
        tm, tb, tr = np.mean(train_losses, axis=0)
        if rank == 0:
            print(f"[Epoch {epoch}] Train   MPJPE={tm:.4f}, Bone={tb:.4f}, Reproj={tr:.2f}")

        # Validation + best-model save
        model.eval()
        val_metrics = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ",
                        disable=(rank!=0), ncols=120)
            for batch in pbar:
                x2d, y3d, k, seg, K_pix, R_pix, t_pix = (
                    batch['x2d'].cuda(local_rank),
                    batch['y3d'].cuda(local_rank),
                    batch['k'].cuda(local_rank),
                    batch['seg'].cuda(local_rank),
                    batch['K_pix'].cuda(local_rank),
                    batch['R_pix'].cuda(local_rank),
                    batch['t_pix'].cuda(local_rank),
                )
                pred      = model(x2d, k, seg)
                loss_m    = mpjpe(pred, y3d)
                bones_p   = bone_lengths(pred, SKELETON_EDGES)
                bones_t   = bone_lengths(y3d,  SKELETON_EDGES)
                loss_b    = (bones_p - bones_t).abs().mean()
                loss_r_px = full_reprojection_loss(pred, x2d, K_pix, R_pix, t_pix)
                loss_r    = loss_r_px / IMG_W
                val_metrics.append((loss_m.item(), loss_b.item(), loss_r.item()))

        vm, vb, vr = np.mean(val_metrics, axis=0)
        if rank == 0:
            print(f"[Epoch {epoch}] Val     MPJPE={vm:.4f}, Bone={vb:.4f}, Reproj={vr:.2f}")
            if vm < best_val:
                best_val = vm
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'best_val': best_val
                }
                if scaler:
                    ckpt['scaler_state_dict'] = scaler.state_dict()
                torch.save(ckpt, f"best_lifter_{D_MODEL}_ddp.pt")
                print(f"→ Saved best model (Val MPJPE={best_val:.4f})")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

