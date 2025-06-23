"""
t-SNE で ChatMusician の embedding と z_p を 2D 可視化し PNG 保存
"""

import h5py, numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

H5 = "data/processed/musiccapz_v2.h5"   # 必要なら v3 に

emb, zp = [], []
with h5py.File(H5, "r") as h5:
    for k in h5.keys():
        emb.append(h5[k]["embedding"][()])
        zp.append(h5[k]["z_p"][()])

X  = np.array(emb)[:, :128]   # 4096→128 に切り出し
ZP = np.array(zp)

data = TSNE(
    n_components=2, init="pca",
    perplexity=min(30, len(X)-1), random_state=0
).fit_transform(np.vstack([X, ZP]))

N = len(X)
plt.figure(figsize=(6,6))
plt.scatter(data[:N,0],  data[:N,1], c="blue",  label="embedding", s=15)
plt.scatter(data[N:,0],  data[N:,1], c="red",   label="z_p",       s=15)
plt.title("t-SNE: caption embedding vs z_p")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_result.png", dpi=150)   # ← PNG 保存
print("✅ tsne_result.png を保存しました")
