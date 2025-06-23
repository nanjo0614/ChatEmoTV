"""
train_mapper.py  (Stage-A 本番学習)
"""

import h5py, torch
from torch.utils.data import TensorDataset, DataLoader
from src.mapper import Mapper
from src.loss   import info_nce

# ── ハイパーパラメータ ─────────────────────────
DS_PATH = "data/processed/musiccapz_v3.h5"   # ← v3 を使用
BATCH   = 64
EPOCHS  = 50
LR      = 1e-4
# ────────────────────────────────────────────

# データ読み込み
emb, zp = [], []
with h5py.File(DS_PATH, "r") as h5:
    for k in h5.keys():
        emb.append(h5[k]["embedding"][()])
        zp.append(h5[k]["z_p"][()])
X = torch.tensor(emb); Y = torch.tensor(zp)
loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)

# モデル
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Mapper().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR)

# Early-Stopping 用
best, wait, patience = float("inf"), 0, 5

for ep in range(1, EPOCHS + 1):
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = info_nce(model(x), y)
        loss.backward(); opt.step()
        total += loss.item()
    avg = total / len(loader)
    print(f"Epoch {ep:02d}  Loss {avg:.4f}")

    # Early-Stopping
    if avg < best:
        best, wait = avg, 0
        torch.save(model.state_dict(), "models/mapper_stageA.pth")
    else:
        wait += 1
        if wait >= patience:
            print("↳ 早期終了：Loss 改善なし")
            break
