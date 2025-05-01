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

# ------- CONFIG -------
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

# ---------- DATASET ----------
class AugmentedNPZDataset(Dataset):
    def __init__(self, npz_dir, window=T):
        self.window = window
        self.index = []
        for npz_path in glob.glob(os.path.join(npz_dir, '*.npz')):
            data = np.load(npz_path)
            P, F, _, _ = data['joints_2d'].shape
            for p in range(P):
                for f_start in range(F - window + 1):
                    self.index.append((npz_path, p, f_start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        npz_path, p, f_start = self.index[idx]
        data = np.load(npz_path)

        x2d = data['joints_2d'][p, f_start:f_start+T]    # (T,J,2)
        y3d = data['joints_3d'][    f_start:f_start+T]  # (T,J,3)
        seg = data['segments_lengths'][f_start]         # (S,1)
        K_pix = data['K'][p]                            # (3,3)

        # normalize intrinsics → k_vec
        f_norm  = K_pix[0,0] / 2000.0
        cx_norm = (K_pix[0,2] - IMG_W/2) / IMG_W
        cy_norm = (K_pix[1,2] - IMG_H/2) / IMG_H
        k_vec = np.array([f_norm, cx_norm, cy_norm], dtype=np.float32)

        # normalize segments & 2D
        seg_vec   = seg.squeeze() / 2.0      # (S,)
        x2d_norm  = x2d.astype(np.float32) / IMG_W

        return {
            'x2d'  : torch.from_numpy(x2d_norm),           
            'y3d'  : torch.from_numpy(y3d.astype(np.float32)),
            'k'    : torch.from_numpy(k_vec),              
            'seg'  : torch.from_numpy(seg_vec),            
            'K_pix': torch.from_numpy(K_pix.astype(np.float32)),
        }

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

# ---------- TRAINING LOOP ----------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load full dataset and split
    full_ds = AugmentedNPZDataset(args.data)
    N = len(full_ds)
    N_val = int(0.1 * N)
    N_train = N - N_val
    train_ds, val_ds = random_split(full_ds, [N_train, N_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # model, optimizer, scheduler
    model = TransformerLifter().to(device)
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt)

    opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
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

            pred = model(x2d, k, seg)

            # compute losses
            loss_m = mpjpe(pred, y3d)
            bones_pred = bone_lengths(pred, SKELETON_EDGES)
            bones_gt   = bone_lengths(y3d,  SKELETON_EDGES)
            loss_b = (bones_pred - bones_gt).abs().mean()
            loss_r_px = reprojection_loss(pred, x2d, K_pix)            # pixels
            loss_r    = loss_r_px / IMG_W                              # normalised

            loss = loss_m + 0.1*loss_b + 0.0*loss_r

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
            torch.save(model.state_dict(), 'best_lifter.pt')
            print(f"→ Saved best model (Val MPJPE={best_val:.4f} m)")

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
