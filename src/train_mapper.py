"""
train_mapper.py  (Batch 学習版)
"""

import h5py, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.mapper import Mapper
from src.loss import info_nce

BATCH = 32
EPOCHS = 10
DS_PATH = "data/processed/musiccapz_v2.h5"

# ── データ読み込み ──────────────────────────
emb, zp = [], []
with h5py.File(DS_PATH, "r") as h5:
    for k in h5.keys():
        emb.append(h5[k]["embedding"][()])
        zp.append(h5[k]["z_p"][()])
X = torch.tensor(emb)  # [N,4096]
Y = torch.tensor(zp)   # [N,128]
loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)

# ── モデル & オプティマイザ ─────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Mapper().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ── 学習ループ ─────────────────────────────
for ep in range(1, EPOCHS + 1):
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = info_nce(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {ep:02d}  Loss {total/len(loader):.4f}")
