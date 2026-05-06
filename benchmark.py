import torch
import torch.nn.functional as F
from qflash import qflash

# (name, Z, H, N, D) -- per-sample shapes (Z is #windows for Swin)
WORKLOADS = [
    ('A1 ViT-T',    1,   3, 197, 64),
    ('A2 ViT-S',    1,   6, 197, 64),
    ('A3 ViT-B',    1,  12, 197, 64),
    ('A4 Swin-1', 64,   3,  49, 32),
    ('A5 Swin-2', 16,   6,  49, 32),
    ('A6 Swin-3',  4,  12,  49, 32),
    ('A7 Swin-4',  1,  24,  49, 32),
]
BATCH_SIZES = [1, 8]

torch.manual_seed(199)
print(f"{'workload':<11} {'SQNR (dB)':>10}", end='')
for b in BATCH_SIZES:
    print(f"  {f'b={b} (ms)':>10}", end='')
print()

for name, Z, H, N, D in WORKLOADS:
    qkv = torch.rand(3, Z, H, N, D, device='cuda', dtype=torch.float16) * 8 - 4
    q, k, v = qkv[0], qkv[1], qkv[2]
    out = qflash.forward(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    sqnr = 20 * torch.log10(torch.norm(ref) / torch.norm(out - ref))
    print(f"{name:<11} {sqnr.item():>10.2f}", end='')

    for b in BATCH_SIZES:
        qkv_b = torch.rand(3, Z * b, H, N, D, device='cuda', dtype=torch.float16) * 8 - 4
        ms = qflash.forward(qkv_b[0], qkv_b[1], qkv_b[2], return_mode='latency')
        print(f"  {ms:>10.4f}", end='')
    print()
