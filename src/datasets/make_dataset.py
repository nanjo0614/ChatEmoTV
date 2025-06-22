"""
正式版 make_dataset.py
- data/raw/*.abc を読み取り
- ChatMusician で caption + embedding
- encode_latent.get_latents で z_p / z_b（ダミー版でOK）
- HDF5 に保存
"""

import argparse
from pathlib import Path
import h5py, numpy as np
from tqdm import tqdm

from src.utils.embed import ChatMusicianEmbedder
from src.utils.caption import Captioner
from src.utils.encode_latent import get_latents


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="ChatMusician ckpt")
    ap.add_argument("--input_dir", default="data/raw", help="ABC/MIDI dir")
    ap.add_argument("--output_file", default="data/processed/musiccapz_v2.h5")
    return ap


def main():
    args = build_parser().parse_args()

    embedder = ChatMusicianEmbedder(args.model_dir)
    captioner = Captioner(args.model_dir)

    files = sorted(Path(args.input_dir).glob("*.abc"))
    assert files, f"No .abc found in {args.input_dir}"

    with h5py.File(args.output_file, "w") as h5:
        for fp in tqdm(files, desc="creating dataset"):
            score = fp.read_text()

            caption = captioner.caption(score)              # ① キャプション生成
            emb = embedder.encode(caption).numpy()          # ② 埋め込み
            z_p, z_b = get_latents(score)                   # ③ ダミー潜在

            grp = h5.create_group(fp.stem)
            grp.create_dataset("score", data=score.encode("utf-8"))
            grp.create_dataset("caption", data=caption.encode("utf-8"))
            grp.create_dataset("embedding", data=emb)
            grp.create_dataset("z_p", data=z_p)
            grp.create_dataset("z_b", data=z_b)

    print(f"✅ {len(files)} 曲を {args.output_file} に保存しました")


if __name__ == "__main__":
    main()
