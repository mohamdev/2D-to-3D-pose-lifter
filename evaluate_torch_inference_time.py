import os
import time
import argparse
import numpy as np
import torch

# Import your model and utilities (ensure train script has no top-level parsing)
from train_lifter_ddp_dynamic_cams_tiny import (
    TransformerLifter,
    sample_direction,
    look_at_R,
    IMG_W, IMG_H, F_MIN, F_MAX, SEG_NORM
)

# Skeleton pairs for segment lengths
SEG_PAIRS = [(2,4),(3,5),(0,1),(0,6),(6,8),(7,9)]


def sample_camera_and_segs(j3d: np.ndarray):
    coords = j3d.reshape(-1,3)
    R_body = np.linalg.norm(coords,axis=1).max()
    min_z   = coords[:,2].min()
    min_xy  = R_body * (1.5/2)
    # extrinsic
    while True:
        v = sample_direction()
        d = np.random.uniform(1.5, 2.5)
        cam_pos = v * R_body * d
        if cam_pos[2] < min_z or np.linalg.norm(cam_pos[:2]) < min_xy:
            continue
        R_w2c = look_at_R(cam_pos)
        t_w2c = -R_w2c.dot(cam_pos)
        break
    # intrinsics
    f  = np.random.uniform(F_MIN, F_MAX)
    cx = IMG_W/2 + np.random.uniform(-0.03,0.03)*IMG_W
    cy = IMG_H/2 + np.random.uniform(-0.03,0.03)*IMG_H
    K_pix = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)
    # segments
    j0 = j3d[0]
    segs = np.array([np.linalg.norm(j0[i]-j0[j]) for i,j in SEG_PAIRS],
                    dtype=np.float32) / SEG_NORM
    return K_pix, R_w2c, t_w2c, segs


def make_example_inputs(j3d: np.ndarray, window: int):
    # sample camera and segments
    K, R, t, segs = sample_camera_and_segs(j3d)
    # build one window
    win3d = j3d[:window]
    X = win3d.reshape(-1,3).T
    Xc = R @ X + t[:,None]
    uvw = K @ Xc
    uv  = (uvw[:2]/(uvw[2:]+1e-8)).T.reshape(window, j3d.shape[1], 2) / IMG_W

    x2d = torch.from_numpy(uv.astype(np.float32)).unsqueeze(0)
    # normalized intrinsics
    f_norm  = K[0,0]/2000.0
    cx_norm = (K[0,2] - IMG_W/2)/IMG_W
    cy_norm = (K[1,2] - IMG_H/2)/IMG_H
    k_vec = torch.tensor([f_norm, cx_norm, cy_norm], dtype=torch.float32).unsqueeze(0)
    seg_vec = torch.from_numpy(segs).unsqueeze(0)
    return x2d, k_vec, seg_vec


def benchmark_model(model: torch.nn.Module, inputs: tuple, device: torch.device, warmup: int = 10, runs: int = 100):
    # Warm-up
    for _ in range(warmup):
        if device.type == 'cuda': torch.cuda.synchronize()
        with torch.inference_mode():
            model(*inputs)
        if device.type == 'cuda': torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            model(*inputs)
        if device.type == 'cuda': torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # µs

    times = np.array(times, dtype=np.float32)
    return times


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch inference with torch.compile on CPU and GPU"
    )
    parser.add_argument('--npz',        required=True,
                        help="Path to .npz with 'joints_3d'")
    parser.add_argument('--checkpoint', required=True,
                        help="Path to your .pt checkpoint")
    parser.add_argument('--window',     type=int, default=13,
                        help="Sliding window size")
    parser.add_argument('--warmup',     type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument('--runs',       type=int, default=100,
                        help="Number of timed runs")
    args = parser.parse_args()

    # load motion clips
    data = np.load(args.npz)
    if 'joints_3d' not in data:
        raise ValueError(".npz must contain 'joints_3d'")
    j3d = data['joints_3d']

    # prepare example inputs on CPU
    x2d_cpu, k_cpu, seg_cpu = make_example_inputs(j3d, args.window)

    # load and eval model
    model = TransformerLifter().eval()
    ck = torch.load(args.checkpoint, map_location='cpu')
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)

    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

    print("\n=== Benchmarking with torch.compile ===")

    # CPU benchmark
    cpu_device = torch.device('cpu')
    model_cpu = model.to(cpu_device)
    compiled_cpu = torch.compile(model_cpu)
    inputs_cpu = (x2d_cpu.to(cpu_device), k_cpu.to(cpu_device), seg_cpu.to(cpu_device))
    times_cpu = benchmark_model(compiled_cpu, inputs_cpu, cpu_device, args.warmup, args.runs)
    print(f"CPU       Avg: {times_cpu.mean():8.2f} µs  Std: {times_cpu.std():6.2f} µs")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        model_gpu = model.to(gpu_device)
        compiled_gpu = torch.compile(model_gpu)
        inputs_gpu = (x2d_cpu.to(gpu_device), k_cpu.to(gpu_device), seg_cpu.to(gpu_device))
        times_gpu = benchmark_model(compiled_gpu, inputs_gpu, gpu_device, args.warmup, args.runs)
        print(f"GPU       Avg: {times_gpu.mean():8.2f} µs  Std: {times_gpu.std():6.2f} µs")
    else:
        print("GPU       not available, skipping.")

if __name__ == '__main__':
    main()
