#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import bisect

print("starting")
#i ------- CONFIG -------
J = 12      # joints
S = 6       # segments
T = 13      # temporal window
D_IN = 11   # 2D + 3 intrinsics + 6 segment lengths
D_MODEL = 128
N_HEAD = 4
N_LAYERS = 4
D_FF = 256
IMG_W, IMG_H = 1920, 1080

SKELETON_EDGES = [
    (0, 2), (2, 4),   # R shoulder→elbow→wrist
    (1, 3), (3, 5),   # L shoulder→elbow→wrist
    (6, 8), (8,10),   # R hip→knee→ankle
    (7, 9), (9,11),   # L hip→knee→ankle
    (0, 1),           # shoulders
    (6, 7),           # hips
    (0, 6), (1, 7),   # torso sides
]


import os, glob, bisect
import numpy as np
import torch
from torch.utils.data import Dataset

class AugmentedNPZDataset(Dataset):
    """
    Memory-frugal dataset for the ‘augmented_npz_grid’ format,
    now with logging so you can see what's happening under the hood.
    """
    def __init__(self,
                 npz_dir: str,
                 window: int = 13,
                 *,
                 img_wh: tuple[int, int] = (1920, 1080),
                 return_extrinsics: bool = True):
        print(f"[Dataset] Initializing from {npz_dir!r} with window={window}")
        self.window            = int(window)
        self.W, self.H         = img_wh
        self.return_extrinsics = bool(return_extrinsics)

        self.meta        = []
        self.cum_windows = []
        total_windows    = 0

        # 1) scan all files, build tiny index
        for path in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
            d = np.load(path, mmap_mode='r')
            P, F, _, _ = d['joints_2d'].shape
            n_f        = max(F - self.window + 1, 0)
            n_w        = P * n_f
            if n_w == 0:
                print(f"[Dataset]  → skip {os.path.basename(path)} (too short)")
                continue

            print(f"[Dataset]  → file {os.path.basename(path)}: P={P}, F={F}, windows={n_w}")
            self.meta.append({
                'path': path,
                'P'   : P,
                'F'   : F,
                'n_f' : n_f,
                'n_w' : n_w,
            })
            self.cum_windows.append(total_windows)
            total_windows += n_w

        self.total_windows = total_windows
        print(f"[Dataset] Done init: {len(self.meta)} files, "
              f"{self.total_windows} total windows.\n")

        if self.total_windows == 0:
            raise RuntimeError(f'No windows found in “{npz_dir}”')

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 2) locate which file & which window
        file_idx = bisect.bisect_right(self.cum_windows, idx) - 1
        meta     = self.meta[file_idx]
        local    = idx - self.cum_windows[file_idx]
        p        = local // meta['n_f']
        f_start  = local %  meta['n_f']
        f_end    = f_start + self.window

        # log the first few mappings
        if idx < 5:
            print(f"[Dataset]  idx={idx} → file#{file_idx} "
                  f"{os.path.basename(meta['path'])}, cam={p}, f_start={f_start}")

        # 3) load & slice
        data = np.load(meta['path'], mmap_mode='r')
        x2d  = data['joints_2d'][p, f_start:f_end]     # (T,J,2)
        y3d  = data['joints_3d'][   f_start:f_end]     # (T,J,3)
        seg  = data['segments_lengths'][f_start]       # (S,1)
        K    = data['K'][p]                            # (3,3)

        # 4) normalise intrinsics → k_vec
        f_norm  =  K[0,0] / 2000.0
        cx_norm = (K[0,2] - self.W/2) / self.W
        cy_norm = (K[1,2] - self.H/2) / self.H
        k_vec   = np.asarray([f_norm, cx_norm, cy_norm], dtype=np.float32)

        # 5) package as torch.Tensors
        sample = {
            'x2d'  : torch.from_numpy(x2d.astype(np.float32) / self.W),  # (T,J,2)
            'y3d'  : torch.from_numpy(y3d.astype(np.float32)),          # (T,J,3)
            'k'    : torch.from_numpy(k_vec),                           # (3,)
            'seg'  : torch.from_numpy(seg.squeeze().astype(np.float32) / 2.0),  # (S,)
            'K_pix': torch.from_numpy(K.astype(np.float32)),            # (3,3)
        }

        if self.return_extrinsics:
            sample['R_pix'] = torch.from_numpy(data['R'][p].astype(np.float32))
            sample['t_pix'] = torch.from_numpy(data['t'][p].astype(np.float32))

        return sample


# ---------- MODEL ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
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
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_FF,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.regressor = nn.Linear(D_MODEL, 3)

    def forward(self, x2d, k, seg):
        B, T, J, _ = x2d.shape
        k_exp   = k[:, None, None, :].expand(B, T, J, 3)
        seg_exp = seg[:, None, None, :].expand(B, T, J, 6)
        tokens  = torch.cat([x2d, k_exp, seg_exp], dim=-1)  # (B,T,J,11)
        tokens  = tokens.reshape(B, T*J, D_IN)
        x       = self.token_embed(tokens)                  # (B,T*J,D_MODEL)

        # special tokens
        k_proj   = self.k_embed(k)       # (B,D_MODEL)
        seg_proj = self.seg_embed(seg)   # (B,D_MODEL)
        k_tok    = self.k_token.expand(B, -1, -1) + k_proj.unsqueeze(1)
        seg_tok  = self.seg_token.expand(B, -1, -1) + seg_proj.unsqueeze(1)
        x        = torch.cat([k_tok, seg_tok, x], dim=1)   # (B,T*J+2,D_MODEL)

        x = self.pos_enc(x)
        out = self.transformer(x)
        out = out[:, 2:, :].reshape(B, T, J, D_MODEL)
        return self.regressor(out)  # (B,T,J,3)

# ---------- LOSS FUNCTIONS ----------
def mpjpe(pred, target):
    return ((pred - target)**2).sum(-1).sqrt().mean()

def bone_lengths(x, pairs):
    return torch.stack([(x[:,:,i]-x[:,:,j]).norm(dim=-1) for (i,j) in pairs], dim=-1)

def reprojection_loss(pred3d, x2d, K):
    B, T, J, _ = pred3d.shape
    pts = pred3d[:,0].reshape(-1,3)      # (B*J,3)
    K_b = K.unsqueeze(1).expand(B, J, 3, 3).reshape(-1,3,3)
    uvw = torch.bmm(K_b, pts.unsqueeze(-1)).squeeze(-1)     # (B*J,3)
    uv  = uvw[:,:2] / (uvw[:,2:3] + 1e-8)
    uv  = uv.reshape(B, J, 2)
    x2d_px = x2d[:,0] * IMG_W        # (B,J,2)
    return (uv - x2d_px).abs().mean()

def full_reprojection_loss(pred3d, x2d, K, R, t):
    B,T,J,_ = pred3d.shape
    pts = pred3d[:,0].reshape(B*J,3)         # (B·J,3)

    R_b = R.unsqueeze(1).expand(B,J,3,3).reshape(-1,3,3)
    t_b = t.unsqueeze(1).expand(B,J,3).reshape(-1,3)

    cam_pts = (R_b @ pts.unsqueeze(-1)).squeeze(-1) + t_b  # (B·J,3)
    K_b     = K.unsqueeze(1).expand(B,J,3,3).reshape(-1,3,3)

    uvw = (K_b @ cam_pts.unsqueeze(-1)).squeeze(-1)
    uv  = uvw[:,:2] / (uvw[:,2:]+1e-8)
    uv  = uv.reshape(B,J,2)

    x2d_px = x2d[:,0] * IMG_W
    return (uv - x2d_px).abs().mean()


# ---------- TRAINING LOOP ----------
def train(args):
    print("start training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("loading dataset")
    # load full dataset and split
    full_ds = AugmentedNPZDataset(args.data)
    N = len(full_ds)
    N_val = int(0.1 * N)
    N_train = N - N_val
    train_ds, val_ds = random_split(full_ds, [N_train, N_val])
    
    print("dataset loaded, configuring loaders")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)
    print("data loaders configurd, instanciating Trnsformer")

    # model, optimizer, scheduler
    model = TransformerLifter().to(device)
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt)

    opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    n_lifters = 0
    print("starting learning loop")
    for epoch in range(1, args.epochs+1):
        # --- TRAIN ---
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            x2d   = batch['x2d'].to(device)
            y3d   = batch['y3d'].to(device)
            k     = batch['k'].to(device)
            seg   = batch['seg'].to(device)
            K_pix = batch['K_pix'].to(device)
            R_pix = batch['R_pix'].to(device)
            t_pix = batch['t_pix'].to(device)

            pred = model(x2d, k, seg)

            # compute losses
            loss_m = mpjpe(pred, y3d)
            bones_pred = bone_lengths(pred, SKELETON_EDGES)
            bones_gt   = bone_lengths(y3d,  SKELETON_EDGES)
            loss_b = (bones_pred - bones_gt).abs().mean()
            loss_r_px = full_reprojection_loss(pred, x2d, K_pix, R_pix, t_pix)            # pixels
            loss_r    = loss_r_px / IMG_W                              # normalised

            loss = loss_m + 1.0*loss_b + 0.0*loss_r

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append((loss_m.item(), loss_b.item(), loss_r.item()))
            pbar.set_postfix({
                'MPJPE': f"{loss_m.item():.4f}",
                'Bone' : f"{loss_b.item():.4f}",
                'Reproj(px)': f"{loss_r_px.item():.0f}",
                'Reproj(n)':  f"{loss_r.item():.4f}",
            })

        sched.step()
        tm, tb, tr = np.mean(train_losses, axis=0)
        print(f"[Epoch {epoch}] Train   MPJPE={tm:.4f} m, Bone={tb:.4f}, Reproj={tr:.2f}")

        # --- VALIDATION ---
        model.eval()
        val_metrics = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
            for batch in pbar:
                x2d   = batch['x2d'].to(device)
                y3d   = batch['y3d'].to(device)
                k     = batch['k'].to(device)
                seg   = batch['seg'].to(device)
                K_pix = batch['K_pix'].to(device)

                pred = model(x2d, k, seg)

                loss_m = mpjpe(pred, y3d)
                bones_pred = bone_lengths(pred, SKELETON_EDGES)
                bones_gt   = bone_lengths(y3d,  SKELETON_EDGES)
                loss_b = (bones_pred - bones_gt).abs().mean()

                loss_r_px = reprojection_loss(pred, x2d, K_pix)            # pixels
                loss_r    = loss_r_px / IMG_W                              # normalised

                val_metrics.append((loss_m.item(), loss_b.item(), loss_r.item()))

        vm, vb, vr = np.mean(val_metrics, axis=0)
        print(f"[Epoch {epoch}] Val     MPJPE={vm:.4f} m, Bone={vb:.4f}, Reproj={vr:.2f}")

        # save best
        if vm < best_val:
            best_val = vm
            torch.save(model.state_dict(), f"best_lifter{n_lifters}.pt")
            print(f"→ Saved best model (Val MPJPE={best_val:.4f} m)")
            n_lifters=n_lifters+1

    print("Training done.")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   type=str, required=True,
                        help="Path to augmented_npz_grid/")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch',  type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a model checkpoint to resume training from")
    args = parser.parse_args()
    train(args)
